import sys
import time
from itertools import accumulate
from math import sqrt
from operator import mul

try:
    from math import sumprod
except ImportError:
    def sumprod(a, b):
        return sum(map(mul, a, b))

N = 5940
T = 81
TP = 75
CHIP = 3
NF = 11
NB = 55
PILOT = [1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0]
NP = len(PILOT)
NS = NP + NF + NB
LB = T + 19
E = LB - T
AP = 10
HEAD = 1 + AP
PARITY = [0b11100, 0b11010, 0b11001, 0b10110, 0b10101, 0b10011]
START = time.time()
REFIT_SECONDS = 2.5
REFIT_LIMIT = 6
KL_ROUNDS = 3


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


CHIPS = prbs(TP // CHIP, 7)
PATTERN = [c for c in CHIPS for _ in range(CHIP)]
SCR_F = prbs(NF, 11)
SCR_B = prbs(NB, 23)
CODES = [field_code(length) for length in range(1, NB + 1)]

DRIVE = [2048.0, 1024.0]
MODEL = None
FIELD_ACC = [0.0] * NF
REFITS = 0


def frame_symbols(bits):
    length = len(bits)
    code = CODES[length - 1]
    field = [code[i] * SCR_F[i] for i in range(NF)]
    payload = [(1.0 if bits[j % length] == '1' else -1.0) * SCR_B[j] for j in range(NB)]
    return PILOT + field + payload


def analyse_baseline(h):
    n = len(h)
    if n < 200:
        return
    tail = h[-(n // 5):]
    base = sum(tail) / len(tail)
    dev = [v - base for v in h[:N]]
    peak = max(map(abs, dev))
    if peak < 4.0:
        return
    floor = max(0.05 * peak, 1.5)
    last = max(i for i, v in enumerate(dev) if abs(v) > floor)
    span = min(len(dev), last + 20)
    step = list(accumulate(dev[:span]))
    smax = max(map(abs, step))
    step += [step[-1]] * (N - span)
    edges = [CHIPS[0]] + [CHIPS[i] - CHIPS[i - 1] for i in range(1, len(CHIPS))] + [-CHIPS[-1]]
    unit = [0.0] * N
    for i, e in enumerate(edges):
        if e:
            off = i * CHIP
            unit[off:] = [p + e * q for p, q in zip(unit[off:], step)]
    cmax = 0.0
    for trial in range(2):
        total = [0.0] * N
        for k, s in enumerate(PILOT + prbs(NS - NP, 99 + trial)):
            off = k * T
            total[off:] = [p + s * q for p, q in zip(total[off:], unit)]
        cmax = max(cmax, max(total), -min(total))
    if cmax <= 0.0:
        return
    if min(h) >= 0 and max(h) <= 4095:
        room = min(base, 4095.0 - base)
        target = 0.5 * room
        ceiling = 0.9 * room
    else:
        target = 2.0 * peak
        ceiling = 4.0 * peak
    target = max(target, 0.5 * peak)
    ceiling = max(ceiling, target)
    bias = 0.5 * smax
    if bias + 0.25 * target <= ceiling:
        amp = min(2047.0, target * 2047.0 / (1.25 * cmax), (ceiling - bias) * 4095.0 / (1.25 * cmax))
        center = 2048.0
    else:
        amp = min(2047.0, target * 4095.0 / (smax + 1.25 * cmax))
        center = amp
    amp = max(amp, 48.0)
    DRIVE[:] = [min(max(center, amp), 4095.0 - amp), amp]


def encode(message):
    bits = ''.join(ch for ch in message if ch in '01')[:NB] or '0'
    center, amp = DRIVE
    waves = {}
    for s in (1.0, -1.0):
        waves[s] = [min(4095, max(0, int(round(center + amp * s * p)))) for p in PATTERN]
    out = [int(round(center))] * N
    for k, s in enumerate(frame_symbols(bits)):
        out[k * T:k * T + TP] = waves[s]
    return out


class Recording:
    def __init__(self, values):
        ref = float(values[0])
        self.v = [float(x) - ref for x in values]
        self.pad = None

    def prepare(self):
        if self.pad is None:
            pad = [0.0] * AP + self.v
            self.pad = pad
            self.pq = [list(accumulate(map(mul, pad, pad[d:]), initial=0.0)) for d in range(AP + 1)]
            self.py = list(accumulate(pad, initial=0.0))

    def replace(self, index, value):
        self.v[index] = value
        self.pad = None


def cholesky(m):
    low = []
    for i in range(len(m)):
        row = m[i]
        cur = [0.0] * (i + 1)
        for j in range(i):
            prev = low[j]
            cur[j] = (row[j] - sumprod(cur[:j], prev[:j])) / prev[j]
        d = row[i] - sumprod(cur[:i], cur[:i])
        if not d > 0.0:
            return None
        cur[i] = sqrt(d)
        low.append(cur)
    return low


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


def factor(m):
    ridge = 1e-9
    for _ in range(4):
        damped = [row[:] for row in m]
        for i in range(len(m)):
            damped[i][i] = damped[i][i] * (1.0 + ridge) + ridge
        low = cholesky(damped)
        if low is not None:
            return low
        ridge *= 100.0
    return None


def symmetrize(m):
    for i in range(len(m)):
        for j in range(i):
            m[i][j] = m[j][i]


def fit_channel(rec, sym, known, slots):
    rec.prepare()
    full = slots > NS
    count = NS if full else slots
    spans = []
    groups = [[], [], []]

    def add(lo, hi):
        if spans and spans[-1][1] == lo:
            spans[-1][1] = hi
        else:
            spans.append([lo, hi])

    for n in range(count):
        if not known[n]:
            continue
        if n and known[n - 1]:
            groups[0].append(n)
            groups[2].append(n - 1)
            add(n * T, n * T + E)
        groups[1].append(n)
        add(n * T + E, (n + 1) * T)
    if full:
        if known[NS - 1]:
            groups[2].append(NS - 1)
            add(NS * T, NS * T + E)
        add(NS * T + E, N)

    width = AP + LB
    pad, pq, py = rec.pad, rec.pq, rec.py
    acc = []
    for group in groups:
        total = [0.0] * width
        for n in group:
            s = sym[n]
            total = [p + s * q for p, q in zip(total, pad[n * T:n * T + width])]
        acc.append(total)

    size = AP + LB + 1
    last = size - 1
    m = [[0.0] * size for _ in range(size)]
    r = [0.0] * size
    rows = 0
    syy = 0.0
    for lo, hi in spans:
        rows += hi - lo
        syy += pq[0][hi + AP] - pq[0][lo + AP]
        r[last] += py[hi + AP] - py[lo + AP]
        for i in range(1, AP + 1):
            r[i - 1] += pq[i][hi + AP - i] - pq[i][lo + AP - i]
            m[i - 1][last] += py[hi + AP - i] - py[lo + AP - i]
            for j in range(i, AP + 1):
                m[i - 1][j - 1] += pq[j - i][hi + AP - j] - pq[j - i][lo + AP - j]
    m[last][last] = float(rows)
    cross = sum(sym[n] * sym[n - 1] for n in groups[0])
    sums = [sum(sym[n] for n in group) for group in groups]
    for j in range(LB):
        g = 0 if j < E else (1 if j < T else 2)
        col = AP + j
        vec = acc[g]
        m[col][col] = float(len(groups[g]))
        m[col][last] = sums[g]
        r[col] = vec[j + AP]
        for i in range(1, AP + 1):
            m[i - 1][col] = vec[j - i + AP]
        if j < E:
            m[col][col + T] = cross
    symmetrize(m)

    low = factor(m)
    if low is None:
        return None
    theta = chol_solve(low, r)
    for _ in range(KL_ROUNDS):
        cost = syy - 2.0 * sumprod(theta, r) + sumprod(theta, [sumprod(row, theta) for row in m])
        shift = max(cost, 0.0) / (1.0 + sumprod(theta[:AP], theta[:AP]))
        if shift <= 0.0:
            break
        moved = [row[:] for row in m]
        for i in range(AP):
            moved[i][i] -= shift
        low = cholesky(moved)
        if low is None:
            break
        theta = chol_solve(low, r)
    return theta[:AP], theta[AP:AP + LB]


def recursion(a, drive):
    out = [0.0] * AP + drive
    rev = a[::-1]
    for t in range(AP, len(out)):
        out[t] += sumprod(rev, out[t - AP:t])
    return out[AP:]


def bounded(g):
    head = max(map(abs, g[:N // 4]))
    tail = max(map(abs, g[-(N // 4):]))
    return head < 1e150 and tail <= 2.0 * head + 1e-12


def build_model(a, beta):
    drive = beta + [0.0] * (N - LB)
    g = None
    for _ in range(8):
        try:
            g = recursion(a, drive)
        except OverflowError:
            g = None
        if g is not None and bounded(g):
            break
        g = None
        a = [c * 0.97 ** (i + 1) for i, c in enumerate(a)]
    if g is None:
        a = [0.0] * AP
        g = drive[:]
    free = recursion(a, [1.0] + [0.0] * (N - 1))
    return {'a': a, 'beta': beta, 'g': g, 'free': free, 'low': None, 'noise': 0.0}


def gram(model, count, t_end):
    g, free = model['g'], model['free']
    size = HEAD + count
    m = [[0.0] * size for _ in range(size)]
    m[0][0] = float(t_end)
    pf = list(accumulate(free[:t_end], initial=0.0))
    pg = list(accumulate(g[:t_end], initial=0.0))
    for d in range(AP):
        lag = list(accumulate(map(mul, free, free[d:t_end]), initial=0.0))
        for j in range(AP - d):
            m[1 + j][1 + j + d] = lag[t_end - j - d]
    for j in range(AP):
        m[0][1 + j] = pf[t_end - j]
        m[1 + j][HEAD] = sumprod(free[:t_end - j], g[j:t_end])
        for k in range(1, count):
            off = k * T
            m[1 + j][HEAD + k] = sumprod(free[off - j:t_end - j], g[:t_end - off])
    for k in range(count):
        m[0][HEAD + k] = pg[t_end - k * T]
    for d in range(count):
        lag = list(accumulate(map(mul, g, g[d * T:t_end]), initial=0.0))
        for k in range(count - d):
            m[HEAD + k][HEAD + k + d] = lag[t_end - (k + d) * T]
    symmetrize(m)
    return factor(m)


def project(model, low, v, count, t_end):
    g, free = model['g'], model['free']
    head = v[:t_end]
    r = [sum(head)]
    for j in range(AP):
        r.append(sumprod(v[j:t_end], free[:t_end - j]))
    for k in range(count):
        off = k * T
        r.append(sumprod(v[off:t_end], g[:t_end - off]))
    theta = chol_solve(low, r)
    return {'theta': theta, 'amps': theta[HEAD:], 'rss': sumprod(head, head) - sumprod(theta, r)}


def detect(model, v, count, t_end):
    low = gram(model, count, t_end)
    if low is None:
        return None
    if count == NS:
        model['low'] = low
    return project(model, low, v, count, t_end)


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


def fitted_model(rec, sym, known, slots):
    fit = fit_channel(rec, sym, known, slots)
    return None if fit is None else build_model(*fit)


def full_decode(rec):
    sym = PILOT + [0.0] * (NS - NP)
    known = [True] * NP + [False] * (NS - NP)
    model = fitted_model(rec, sym, known, NP)
    if model is None:
        return None, None
    for count in (14, 28, 56, NS):
        t_end = count * T if count < NS else N
        result = detect(model, rec.v, count, t_end)
        if result is None:
            break
        adopt(sym, known, result['amps'], count)
        model = fitted_model(rec, sym, known, count if count < NS else NS + 1) or model
    result = detect(model, rec.v, NS, N)
    if result is None:
        return None, None
    return refine(rec, model, result, 2)


def refine(rec, model, result, rounds=1):
    sym = PILOT + [0.0] * (NS - NP)
    known = [True] * NP + [False] * (NS - NP)
    for _ in range(rounds):
        if not adopt(sym, known, result['amps'], NS):
            break
        candidate = fitted_model(rec, sym, known, NS + 1)
        if candidate is None:
            break
        outcome = detect(candidate, rec.v, NS, N)
        if outcome is None or outcome['rss'] >= result['rss']:
            break
        model, result = candidate, outcome
    return model, result


def predict(model, theta):
    beta = model['beta']
    drive = [0.0] * (N + LB)
    drive[:AP] = theta[1:HEAD]
    for k, amp in enumerate(theta[HEAD:]):
        off = k * T
        drive[off:off + LB] = [p + amp * b for p, b in zip(drive[off:off + LB], beta)]
    c0 = theta[0]
    return [c0 + x for x in recursion(model['a'], drive[:N])]


def clean(rec, model, theta):
    fit = predict(model, theta)
    res = [x - y for x, y in zip(rec.v, fit)]
    limit = max(6.0 * 1.4826 * sorted(map(abs, res))[N // 2], 2.0)
    bad = [i for i, x in enumerate(res) if abs(x) > limit]
    if not bad or len(bad) > N // 4:
        return False
    for i in bad:
        for j in range(max(0, i - 2), min(N, i + 3)):
            rec.replace(j, fit[j])
    return True


def acceptable(result, model):
    alpha, quality = pilot_gain(result['amps'])
    return quality > 0.3 and result['rss'] / N <= 4.0 * model['noise'] + 1.0


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
    global MODEL, REFITS
    if not values:
        return '0'
    rec = Recording(list(values[:N]) + [values[-1]] * (N - len(values)))
    result = None
    if MODEL is not None:
        result = project(MODEL, MODEL['low'], rec.v, NS, N)
        if not acceptable(result, MODEL):
            if clean(rec, MODEL, result['theta']):
                result = project(MODEL, MODEL['low'], rec.v, NS, N)
            if not acceptable(result, MODEL):
                result = None
    if result is None:
        model, result = full_decode(rec)
        if model is None:
            return '0'
        if clean(rec, model, result['theta']):
            model, result = refine(rec, model, project(model, model['low'], rec.v, NS, N), 2)
        model['noise'] = result['rss'] / N
        MODEL = model if pilot_gain(result['amps'])[1] > 0.3 else None
    elif REFITS < REFIT_LIMIT and time.time() - START < REFIT_SECONDS:
        REFITS += 1
        model, result = refine(rec, MODEL, result)
        if model is not MODEL:
            model['noise'] = result['rss'] / N
            MODEL = model
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
