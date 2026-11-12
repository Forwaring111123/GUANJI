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
ORDERS = (2, 4, 6, 8)
PMAX = max(ORDERS)
PARITY = [0b11100, 0b11010, 0b11001, 0b10110, 0b10101, 0b10011]
START = time.time()
REFIT_SECONDS = 2.5
REFIT_LIMIT = 6


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
            pad = [0.0] * PMAX + self.v
            self.pad = pad
            self.pq = [list(accumulate(map(mul, pad, pad[d:]), initial=0.0)) for d in range(PMAX + 1)]
            self.py = list(accumulate(pad, initial=0.0))

    def replace(self, index, value):
        self.v[index] = value
        self.pad = None


def chol_rows(low, m, stop, ridge, shift=0.0, plain=0):
    low = low[:]
    for i in range(len(low), stop):
        row = m[i]
        cur = [0.0] * (i + 1)
        for j in range(i):
            prev = low[j]
            cur[j] = (row[j] - sumprod(cur[:j], prev[:j])) / prev[j]
        d = row[i] * (1.0 + ridge) + ridge - sumprod(cur[:i], cur[:i])
        if i >= plain:
            d -= shift
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


def factor(m, stop=None):
    ridge = 1e-9
    for _ in range(4):
        low = chol_rows([], m, len(m) if stop is None else stop, ridge)
        if low is not None:
            return low, ridge
        ridge *= 100.0
    return None, ridge


def symmetrize(m):
    for i in range(len(m)):
        for j in range(i):
            m[i][j] = m[j][i]


def assemble(rec, sym, known, slots):
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

    width = PMAX + LB
    pad, pq, py = rec.pad, rec.pq, rec.py
    acc = []
    for group in groups:
        total = [0.0] * width
        for n in group:
            s = sym[n]
            total = [p + s * q for p, q in zip(total, pad[n * T:n * T + width])]
        acc.append(total)

    size = LB + 1 + PMAX
    m = [[0.0] * size for _ in range(size)]
    r = [0.0] * size
    rows = 0
    syy = 0.0
    for lo, hi in spans:
        rows += hi - lo
        syy += pq[0][hi + PMAX] - pq[0][lo + PMAX]
        r[LB] += py[hi + PMAX] - py[lo + PMAX]
        for i in range(1, PMAX + 1):
            r[LB + i] += pq[i][hi + PMAX - i] - pq[i][lo + PMAX - i]
            m[LB][LB + i] += py[hi + PMAX - i] - py[lo + PMAX - i]
            for j in range(i, PMAX + 1):
                m[LB + i][LB + j] += pq[j - i][hi + PMAX - j] - pq[j - i][lo + PMAX - j]
    m[LB][LB] = float(rows)
    cross = sum(sym[n] * sym[n - 1] for n in groups[0])
    sums = [sum(sym[n] for n in group) for group in groups]
    for j in range(LB):
        g = 0 if j < E else (1 if j < T else 2)
        vec = acc[g]
        m[j][j] = float(len(groups[g]))
        m[j][LB] = sums[g]
        r[j] = vec[j + PMAX]
        for i in range(1, PMAX + 1):
            m[j][LB + i] = vec[j - i + PMAX]
        if j < E:
            m[j][j + T] = cross
    symmetrize(m)
    return m, r, syy


def smallest_vector(tri):
    size = len(tri)
    v = [0.0] * (size - 1) + [1.0]
    for _ in range(40):
        z = []
        for i in range(size):
            z.append((v[i] - sumprod(tri[i][:i], z)) / tri[i][i])
        w = [0.0] * size
        for i in range(size - 1, -1, -1):
            acc = z[i]
            for k in range(i + 1, size):
                acc -= tri[k][i] * w[k]
            w[i] = acc / tri[i][i]
        norm = sqrt(sumprod(w, w))
        if not norm > 0.0:
            break
        v = [x / norm for x in w]
    return v


def fit_family(rec, sym, known, slots, orders=ORDERS):
    m, r, syy = assemble(rec, sym, known, slots)
    start = LB + 1
    base, ridge = factor(m, start)
    family = {}
    if base is None:
        return family
    low = base
    for order in sorted(orders):
        n = start + order
        low = chol_rows(low, m, n, ridge)
        if low is None:
            break
        z = []
        for i in range(n):
            z.append((r[i] - sumprod(low[i][:i], z)) / low[i][i])
        spare = syy - sumprod(z, z)
        a = None
        if spare > 1e-12 * syy:
            tri = [low[i][start:] for i in range(start, n)] + [z[start:] + [sqrt(spare)]]
            vec = smallest_vector(tri)
            if abs(vec[-1]) > 1e-9:
                a = [-x / vec[-1] for x in vec[:-1]]
        if a is None:
            a = chol_solve(low, r[:n])[start:]
        rest = [r[j] - sumprod(m[j][start:n], a) for j in range(start)]
        family[order] = (a, chol_solve(base, rest)[:LB])
    return family


def recursion(a, drive):
    order = len(a)
    out = [0.0] * order + drive
    rev = a[::-1]
    for t in range(order, len(out)):
        out[t] += sumprod(rev, out[t - order:t])
    return out[order:]


def bounded(g):
    quarter = max(1, len(g) // 4)
    head = max(map(abs, g[:quarter]))
    tail = max(map(abs, g[-quarter:]))
    return head < 1e150 and tail <= 2.0 * head + 1e-12


def build_model(a, beta, horizon):
    drive = (beta + [0.0] * (N - LB))[:horizon]
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
        a = [0.0] * len(a)
        g = drive[:]
    free = recursion(a, [1.0] + [0.0] * (horizon - 1))
    return {'a': a, 'beta': beta, 'g': g, 'free': free, 'low': None, 'noise': 0.0}


def gram(model, count, t_end):
    g, free = model['g'], model['free']
    order = len(model['a'])
    head = 1 + order
    size = head + count
    m = [[0.0] * size for _ in range(size)]
    m[0][0] = float(t_end)
    pf = list(accumulate(free[:t_end], initial=0.0))
    pg = list(accumulate(g[:t_end], initial=0.0))
    for d in range(order):
        lag = list(accumulate(map(mul, free, free[d:t_end]), initial=0.0))
        for j in range(order - d):
            m[1 + j][1 + j + d] = lag[t_end - j - d]
    for j in range(order):
        m[0][1 + j] = pf[t_end - j]
        m[1 + j][head] = sumprod(free[:t_end - j], g[j:t_end])
        for k in range(1, count):
            off = k * T
            m[1 + j][head + k] = sumprod(free[off - j:t_end - j], g[:t_end - off])
    for k in range(count):
        m[0][head + k] = pg[t_end - k * T]
    for d in range(count):
        lag = list(accumulate(map(mul, g, g[d * T:t_end]), initial=0.0))
        for k in range(count - d):
            m[head + k][head + k + d] = lag[t_end - (k + d) * T]
    symmetrize(m)
    return factor(m)[0]


def project(model, low, v, count, t_end):
    g, free = model['g'], model['free']
    order = len(model['a'])
    head = v[:t_end]
    r = [sum(head)]
    for j in range(order):
        r.append(sumprod(v[j:t_end], free[:t_end - j]))
    for k in range(count):
        off = k * T
        r.append(sumprod(v[off:t_end], g[:t_end - off]))
    theta = chol_solve(low, r)
    return {'theta': theta, 'amps': theta[1 + order:], 'rss': sumprod(head, head) - sumprod(theta, r)}


def detect(model, v, count, t_end):
    low = gram(model, count, t_end)
    if low is None:
        return None
    if count == NS:
        model['low'] = low
    return project(model, low, v, count, t_end)


def choose(rec, family, count, t_end, keep=1):
    ranked = []
    for order, (a, beta) in family.items():
        model = build_model(a, beta, t_end)
        result = detect(model, rec.v, count, t_end)
        if result is not None and result['rss'] == result['rss']:
            ranked.append((pilot_gain(result['amps'])[1] <= 0.3, result['rss'], order, model, result))
    ranked.sort(key=lambda item: item[:3])
    if not ranked:
        return None, None, ()
    return ranked[0][3], ranked[0][4], tuple(item[2] for item in ranked[:keep])


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


def full_decode(rec):
    sym = PILOT + [0.0] * (NS - NP)
    known = [True] * NP + [False] * (NS - NP)
    orders = ORDERS
    family = fit_family(rec, sym, known, NP, orders)
    for count in (14, 28, 56, NS):
        t_end = count * T if count < NS else N
        model, result, orders = choose(rec, family, count, t_end, 4 if count < 28 else 2)
        if model is None:
            return None, None
        adopt(sym, known, result['amps'], count)
        family = fit_family(rec, sym, known, count if count < NS else NS + 1, orders)
    model, result, _ = choose(rec, family, NS, N)
    if model is None:
        return None, None
    return refine(rec, model, result, 2)


def refine(rec, model, result, rounds=1):
    sym = PILOT + [0.0] * (NS - NP)
    known = [True] * NP + [False] * (NS - NP)
    for _ in range(rounds):
        if not adopt(sym, known, result['amps'], NS):
            break
        family = fit_family(rec, sym, known, NS + 1, (len(model['a']),))
        candidate, outcome, _ = choose(rec, family, NS, N)
        if candidate is None or outcome['rss'] >= result['rss']:
            break
        model, result = candidate, outcome
    return model, result


def predict(model, theta):
    beta = model['beta']
    head = 1 + len(model['a'])
    drive = [0.0] * (N + LB)
    drive[:head - 1] = theta[1:head]
    for k, amp in enumerate(theta[head:]):
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
