import sys
from itertools import accumulate
from math import cos, exp, log, pi, sin, sqrt
from operator import mul

try:
    from math import sumprod
except ImportError:
    def sumprod(a, b):
        return sum(map(mul, a, b))

N = 5940
T = 80
NF = 11
NB = 55
PILOT = [1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0]
NP = len(PILOT)
NS = NP + NF + NB
STAGES = ((NP, 4, 14), (14, 5, 28), (28, 7, 56), (56, 10, NS))
TAPS = 12
SHAPE_SLOTS = 4
PARITY = [0b11100, 0b11010, 0b11001, 0b10110, 0b10101, 0b10011]


def prbs(count, seed):
    out = []
    state = seed
    for _ in range(count):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(1.0 if (state >> 16) & 1 else -1.0)
    return out


def field_code(length):
    value = length - 1
    bits = [(value >> (5 - i)) & 1 for i in range(6)]
    parity = 0
    for i in range(6):
        if bits[i]:
            parity ^= PARITY[i]
    bits += [(parity >> (4 - i)) & 1 for i in range(5)]
    return [1.0 if b else -1.0 for b in bits]


SCR_F = prbs(NF, 11)
SCR_B = prbs(NB, 23)
CODES = [field_code(length) for length in range(1, NB + 1)]

CONVENTION = None
SHAPER = [(0, 1.0)]
NOTCH = [(0, 1.0)]
DRIVE = [2048.0, 1024.0, False]
BASE = None
GUARD = None
MODEL = None
FIELD_ACC = [0.0] * NF


def frame_symbols(bits):
    length = len(bits)
    code = CODES[length - 1]
    field = [code[i] * SCR_F[i] for i in range(NF)]
    payload = [(1.0 if bits[j % length] == '1' else -1.0) * SCR_B[j] for j in range(NB)]
    return PILOT + field + payload


def apply_fir(x, taps, pad):
    out = [0.0] * len(x)
    for delay, weight in taps:
        out = [o + weight * v for o, v in zip(out, [pad] * delay + x)]
    return out


def explained(dev, theta, rate, length):
    weights = [exp(-rate * t) for t in range(3, 3 + length)]
    c = [w * cos(theta * t) for t, w in enumerate(weights, 3)]
    d = [w * sin(theta * t) for t, w in enumerate(weights, 3)]
    y = dev[3:3 + length]
    cc, cd, dd = sumprod(c, c), sumprod(c, d), sumprod(d, d)
    det = cc * dd - cd * cd
    if not det > 1e-12 * cc * dd:
        return 0.0, 0.0, 0.0
    yc, yd = sumprod(y, c), sumprod(y, d)
    alpha = (yc * dd - yd * cd) / det
    beta = (yd * cc - yc * cd) / det
    return alpha * yc + beta * yd, alpha, beta


def maximise(func, lo, hi, steps):
    ratio = 0.6180339887498949
    x1 = hi - ratio * (hi - lo)
    x2 = lo + ratio * (hi - lo)
    f1, f2 = func(x1), func(x2)
    for _ in range(steps):
        if f1 < f2:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + ratio * (hi - lo)
            f2 = func(x2)
        else:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - ratio * (hi - lo)
            f1 = func(x1)
    return x1 if f1 > f2 else x2


def find_mode(dev, sigma, share=0.0):
    n = len(dev)
    lags = min(600, n // 4)
    width = n - lags - 2
    seg = dev[2:2 + width]
    corr = [sumprod(seg, dev[2 + k:2 + k + width]) for k in range(1, lags + 1)]
    low = 2.0 * pi / (1.5 * SHAPE_SLOTS * T)
    grid = 300
    best = (-1.0, low)
    for j in range(grid):
        theta = low + (2.9 - low) * j / (grid - 1)
        re = sumprod(corr, [cos(theta * k) for k in range(1, lags + 1)])
        im = sumprod(corr, [sin(theta * k) for k in range(1, lags + 1)])
        if re * re + im * im > best[0]:
            best = (re * re + im * im, theta)
    theta = best[1]
    rates = [1.0 / t for t in (30, 60, 120, 250, 500, 1000, 2000, 4000, 16000)]
    rate = max(rates, key=lambda q: explained(dev, theta, q, min(n - 3, 600))[0])
    for length in (600, 2400, n):
        length = min(n - 3, length, int(6.0 / rate) + 200)
        reach = 0.8 * 2.0 * pi / length
        theta = maximise(lambda x: explained(dev, x, rate, length)[0], max(theta - reach, 1e-3), theta + reach, 10)
        rate = exp(maximise(lambda x: explained(dev, theta, exp(x), length)[0], log(rate / 2.5), log(rate * 2.5), 8))
    length = min(n - 3, int(6.0 / rate) + 200)
    energy, alpha, beta = explained(dev, theta, rate, length)
    window = dev[3:3 + length]
    if energy < max(25.0 * sigma * sigma, share * sumprod(window, window)):
        return None
    if exp(-rate * 2.0 * pi / theta) < 0.05 or not theta < 2.8:
        return None
    return theta, exp(-rate), alpha, beta


def shaper_taps(theta, radius):
    n1 = int(pi / theta)
    n2 = n1 + 1
    k1 = -radius ** n1 * sin(theta * n2) / sin(theta)
    k2 = radius ** n2 * sin(theta * n1) / sin(theta)
    gain = 1.0 + k1 + k2
    return [(d, w / gain) for d, w in ((0, 1.0), (n1, k1), (n2, k2)) if w > 1e-9]


def noise_level(x):
    bends = sorted(abs(x[i] - 2.0 * x[i - 1] + x[i - 2]) for i in range(2, len(x)))
    return max(0.3, 1.4826 * bends[len(bends) // 2] / sqrt(6.0))


def analyse_baseline(h):
    global BASE
    n = len(h)
    if n < 400:
        return
    tail = h[-(n // 5):]
    base = sum(tail) / len(tail)
    sigma = noise_level(tail)
    dev = [v - base for v in h[:N]]
    if max(map(abs, dev)) < max(4.0, 3.0 * sigma):
        return
    mode = find_mode(dev, sigma)
    if mode is not None:
        theta, radius, alpha, beta = mode
        span = min(N, int(-7.0 / log(radius)) + 50) if radius < 1.0 else N
        model = [0.0] + [radius ** t * (alpha * cos(theta * t) + beta * sin(theta * t)) for t in range(1, span)]
        if theta >= 2.0 * pi / (SHAPE_SLOTS * T):
            SHAPER[:] = shaper_taps(theta, radius)
            if SHAPER[-1][0] <= T // 2:
                NOTCH[:] = SHAPER
    else:
        floor = max(0.05 * max(map(abs, dev)), 4.0 * sigma)
        marks = [i for i, v in enumerate(dev) if abs(v) > floor]
        if not marks:
            return
        model = dev[:min(len(dev), marks[-1] + 20)]
    peak = max(map(abs, model))
    step = list(accumulate(model))
    step += [step[-1]] * (N - len(step))
    step = step[:N]
    cmax = 0.0
    for trial in range(2):
        syms = PILOT + prbs(NS - NP, 99 + trial) + [0.0]
        total = [0.0] * N
        prev = 0.0
        for k, s in enumerate(syms):
            if s != prev:
                off = T + k * T
                jump = s - prev
                total[off:] = [p + jump * q for p, q in zip(total[off:], step)]
                prev = s
        total = apply_fir(total, SHAPER, 0.0)
        cmax = max(cmax, max(total), -min(total))
    if cmax <= 0.0:
        return
    if min(h) >= 0 and max(h) <= 4095:
        room = max(min(base, 4095.0 - base), 1.5 * peak)
        side = max((4095.0 - base) if step[-1] > 0.0 else base, 1.5 * peak)
    else:
        room = side = 4.0 * peak
    BASE = {
        'model': model, 'sigma': sigma, 'room': room, 'side': side, 'cmax': 1.25 * cmax,
        'smax': max(map(abs, step)), 'szmax': max(map(abs, apply_fir(step, SHAPER, 0.0))),
    }
    plan_drive()


def plan_drive():
    room, cmax, smax = BASE['room'], BASE['cmax'], BASE['smax']
    one_sided = max(48.0, min(2047.0, 0.6 * BASE['side'] * 4095.0 / (BASE['szmax'] + cmax)))
    two_sided = max(48.0, min(2047.0, 0.35 * room * 2047.0 / cmax))
    if CONVENTION == 'U':
        DRIVE[:] = [one_sided, one_sided, True]
        return
    if CONVENTION == 'B':
        DRIVE[:] = [2048.0, two_sided, False]
        return
    best = (0.0, 2048.0)
    for index in range(64, -1, -1):
        center = 32.0 * index
        amp = min(
            (0.9 * room * 4095.0 - center * smax) / cmax,
            (0.9 * room * 2047.0 - (2048.0 - center) * smax) / cmax,
            two_sided,
            center,
            4095.0 - center,
        )
        if amp > best[0]:
            best = (amp, center)
    if best[0] >= max(48.0, 0.4 * one_sided):
        DRIVE[:] = [best[1], best[0], False]
    else:
        DRIVE[:] = [one_sided, one_sided, True]


def detect_convention(raw):
    global CONVENTION
    h = BASE['model'] + [0.0] * T
    seen = [x - raw[0] for x in raw[:T]]
    error_u = error_b = 0.0
    for t in range(T):
        back = h[t::-1]
        u = sumprod(GUARD[:t + 1], back)
        error_u += (seen[t] - u / 4095.0) ** 2
        error_b += (seen[t] - (u - 2048.0 * sum(back)) / 2047.0) ** 2
    margin = 25.0 * T * BASE['sigma'] ** 2
    if error_u < 0.25 * error_b and error_b - error_u > margin:
        CONVENTION = 'U'
    elif error_b < 0.25 * error_u and error_u - error_b > margin:
        CONVENTION = 'B'
    if CONVENTION is not None:
        plan_drive()


def encode(message):
    global GUARD
    bits = ''.join(ch for ch in message if ch in '01')[:NB] or '0'
    center, amp, shaped_onset = DRIVE
    levels = [0.0] * N
    for k, s in enumerate(frame_symbols(bits)):
        levels[T + k * T:T + (k + 1) * T] = [s] * T
    if shaped_onset:
        wave = apply_fir([center + amp * x for x in levels], SHAPER, 0.0)
    else:
        wave = [center + amp * x for x in apply_fir(levels, SHAPER, 0.0)]
    out = [min(4095, max(0, int(round(x)))) for x in wave]
    GUARD = [float(x) for x in out[:T]]
    return out


def cholesky(m):
    for ridge in (1e-9, 1e-7, 1e-5, 1e-3):
        low = []
        for i in range(len(m)):
            row = m[i]
            cur = [0.0] * (i + 1)
            for j in range(i):
                prev = low[j]
                cur[j] = (row[j] - sumprod(cur[:j], prev[:j])) / prev[j]
            d = row[i] * (1.0 + ridge) + ridge - sumprod(cur[:i], cur[:i])
            if not d > 0.0:
                low = None
                break
            cur[i] = sqrt(d)
            low.append(cur)
        if low is not None:
            return low
    return None


def chol_solve(low, b):
    n = len(b)
    z = [0.0] * n
    for i in range(n):
        row = low[i]
        z[i] = (b[i] - sumprod(row[:i], z[:i])) / row[i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        acc = z[i]
        for k in range(i + 1, n):
            acc -= low[k][i] * x[k]
        x[i] = acc / low[i][i]
    return x


def estimate(v, sym, known, count, taps):
    size = taps + 1
    q = [[0.0] * size for _ in range(size)]
    acc = [[0.0] * T for _ in range(size)]
    used = 0
    for n in range(1, count + 1):
        first = max(0, n - taps)
        if not all(known[first:n]):
            continue
        z = [1.0] + [sym[n - 1 - m] if n - 1 - m >= 0 else 0.0 for m in range(taps)]
        seg = v[n * T:(n + 1) * T]
        used += 1
        for i in range(size):
            zi = z[i]
            if zi:
                acc[i] = [p + zi * x for p, x in zip(acc[i], seg)]
                for j in range(size):
                    q[i][j] += zi * z[j]
    if used < size:
        return None
    low = cholesky(q)
    if low is None:
        return None
    g = [0.0] * (taps * T)
    for phase in range(T):
        x = chol_solve(low, [acc[i][phase] for i in range(size)])
        for m in range(taps):
            g[m * T + phase] = x[1 + m]
    return g


def gram(g, count, t_end):
    length = len(g)
    taps = length // T
    size = count + 1
    m = [[0.0] * size for _ in range(size)]
    m[0][0] = float(t_end - T)
    pg = list(accumulate(g, initial=0.0))
    lags = [list(accumulate(map(mul, g, g[d * T:]), initial=0.0)) for d in range(taps)]
    for k in range(count):
        span = min(length, t_end - T - k * T)
        m[0][1 + k] = m[1 + k][0] = pg[span]
        for d in range(taps):
            other = k + d
            if other >= count:
                break
            value = lags[d][max(0, min(length - d * T, t_end - T - other * T))]
            m[1 + k][1 + other] = m[1 + other][1 + k] = value
    return cholesky(m)


def project(g, low, v, count, t_end):
    length = len(g)
    body = v[T:t_end]
    r = [sum(body)]
    for k in range(count):
        start = T + k * T
        span = min(length, t_end - start)
        r.append(sumprod(v[start:start + span], g[:span]))
    theta = chol_solve(low, r)
    return {'theta': theta, 'amps': theta[1:], 'rss': sumprod(body, body) - sumprod(theta, r)}


def detect(g, v, count, t_end):
    low = gram(g, count, t_end)
    return None if low is None else project(g, low, v, count, t_end)


def pilot_gain(amps):
    scores = [amps[k] * PILOT[k] for k in range(NP)]
    alpha = sum(scores) / NP
    return alpha, (min(scores) / alpha if alpha > 0.0 else -1.0)


def adopt(sym, known, amps, count):
    alpha, _ = pilot_gain(amps)
    changed = False
    for k in range(NP, count):
        value = 1.0 if amps[k] > 0.0 else -1.0
        sure = alpha > 0.0 and abs(amps[k]) > 0.35 * alpha
        if value != sym[k] or sure != known[k]:
            changed = True
        sym[k], known[k] = value, sure
    return changed


def refine(v, g, result, rounds):
    sym = PILOT + [0.0] * (NS - NP)
    known = [True] * NP + [False] * (NS - NP)
    for _ in range(rounds):
        if not adopt(sym, known, result['amps'], NS):
            break
        candidate = estimate(v, sym, known, NS, TAPS)
        if candidate is None:
            break
        outcome = detect(candidate, v, NS, N)
        if outcome is None or outcome['rss'] >= result['rss']:
            break
        g, result = candidate, outcome
    return g, result


def full_decode(v):
    sym = PILOT + [0.0] * (NS - NP)
    known = [True] * NP + [False] * (NS - NP)
    g = None
    for have, taps, count in STAGES:
        candidate = estimate(v, sym, known, have, taps)
        if candidate is not None:
            g = candidate
        if g is None:
            return None, None
        result = detect(g, v, count, min(N, (count + 1) * T) if count < NS else N)
        if result is None:
            return None, None
        adopt(sym, known, result['amps'], count)
    return refine(v, g, result, 3)


def predict(g, theta):
    out = [theta[0]] * (N + len(g))
    for k, amp in enumerate(theta[1:]):
        start = T + k * T
        out[start:start + len(g)] = [p + amp * x for p, x in zip(out[start:start + len(g)], g)]
    return out[:N]


def clean(v, g, theta):
    fit = predict(g, theta)
    res = [x - y for x, y in zip(v[T:], fit[T:])]
    limit = max(6.0 * 1.4826 * sorted(map(abs, res))[len(res) // 2], 2.0)
    bad = [T + i for i, x in enumerate(res) if abs(x) > limit]
    if not bad or len(bad) > N // 4:
        return False
    for i in bad:
        for j in range(max(T, i - 2), min(N, i + 3)):
            v[j] = fit[j]
    return True


def extra_notch(v, g, result):
    alpha = pilot_gain(result['amps'])[0]
    if alpha <= 0.0 or result['rss'] / N < 0.01 * alpha * alpha * sumprod(g, g) / T:
        return None
    fit = predict(g, result['theta'])
    mode = find_mode([x - y for x, y in zip(v[T:], fit[T:])], 0.0, 0.2)
    if mode is None or mode[0] < 2.0 * pi / (SHAPE_SLOTS * T):
        return None
    theta = mode[0]
    taps = shaper_taps(theta, max(mode[1], 0.9 ** (theta / pi)))
    moved = apply_fir(v, taps, 0.0)
    g2, result2 = full_decode(moved)
    if g2 is None:
        return None
    alpha2, quality2 = pilot_gain(result2['amps'])
    if quality2 <= 0.3 or result2['rss'] / (alpha2 * alpha2) >= 0.85 * result['rss'] / (alpha * alpha):
        return None
    return taps, moved, g2, result2


def combine(first, second):
    merged = {}
    for d1, w1 in first:
        for d2, w2 in second:
            merged[d1 + d2] = merged.get(d1 + d2, 0.0) + w1 * w2
    return sorted(merged.items())


def acceptable(result):
    return pilot_gain(result['amps'])[1] > 0.3 and result['rss'] / N <= 4.0 * MODEL['noise'] + 1.0


def read_message(amps):
    alpha, quality = pilot_gain(amps)
    if alpha <= 0.0:
        alpha = sum(map(abs, amps)) / len(amps) or 1.0
    soft = [max(-1.5, min(1.5, a / alpha)) for a in amps]
    field = [soft[NP + i] * SCR_F[i] for i in range(NF)]
    data = [soft[NP + NF + j] * SCR_B[j] for j in range(NB)]
    best = None
    for length in range(1, NB + 1):
        sums = [sum(data[i::length]) for i in range(length)]
        score = sum(map(abs, sums))
        score += sum((f + past) * c for f, past, c in zip(field, FIELD_ACC, CODES[length - 1]))
        if best is None or score > best[0]:
            best = (score, sums)
    if quality > 0.5:
        for i in range(NF):
            FIELD_ACC[i] = max(-20.0, min(20.0, FIELD_ACC[i] + field[i]))
    return ''.join('1' if s > 0.0 else '0' for s in best[1])


def decode(values):
    global MODEL
    if not values:
        return '0'
    raw = [float(x) for x in values[:N]] + [float(values[-1])] * (N - len(values))
    v = [x - raw[0] for x in apply_fir(raw, NOTCH, raw[0])]
    g = result = None
    notch = None
    if MODEL is not None:
        notch = MODEL['notch']
        w = apply_fir(v, notch, 0.0) if notch else v
        result = project(MODEL['g'], MODEL['low'], w, NS, N)
        if not acceptable(result) and clean(w, MODEL['g'], result['theta']):
            result = project(MODEL['g'], MODEL['low'], w, NS, N)
        if acceptable(result):
            v = w
            g, result = refine(v, MODEL['g'], result, 1)
        else:
            result = notch = None
    if result is None:
        g, result = full_decode(v)
        if g is None:
            return '0'
        for _ in range(3):
            found = extra_notch(v, g, result)
            if found is None:
                break
            notch = combine(notch, found[0]) if notch else found[0]
            v, g, result = found[1:]
    if clean(v, g, result['theta']):
        again = detect(g, v, NS, N)
        if again is not None:
            g, result = refine(v, g, again, 2)
    if pilot_gain(result['amps'])[1] > 0.3:
        if MODEL is None or g is not MODEL['g']:
            MODEL = {'g': g, 'low': gram(g, NS, N), 'noise': result['rss'] / N, 'notch': notch}
    else:
        MODEL = None
    if CONVENTION is None and BASE is not None and GUARD is not None:
        detect_convention(raw)
    return read_message(result['amps'])


def read_integers(line):
    out = []
    for token in line.split():
        try:
            out.append(int(token))
        except ValueError:
            try:
                out.append(int(float(token)))
            except ValueError:
                pass
    return out


def main():
    stream = sys.stdin
    out = sys.stdout

    def next_line():
        while True:
            line = stream.readline()
            if not line:
                return None
            if line.strip():
                return line.strip()

    while True:
        line = next_line()
        if line is None:
            break
        parts = line.split(None, 1)
        mode = parts[0]
        if mode == '0':
            break
        if mode not in ('1', '2', '3'):
            continue
        payload = parts[1] if len(parts) > 1 else next_line()
        if payload is None:
            break
        if mode == '1':
            out.write(' '.join(map(str, encode(payload))) + '\n')
            out.flush()
        elif mode == '2':
            out.write(decode(read_integers(payload)) + '\n')
            out.flush()
        else:
            analyse_baseline(read_integers(payload))


if __name__ == '__main__':
    main()
