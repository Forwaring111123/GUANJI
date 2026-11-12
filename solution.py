import sys

try:
    import numpy as np
except Exception:
    np = None

N = 5940
T = 88
NL = 6
MAXB = 55
PILOT = [1.0, 1.0, -1.0, 1.0, -1.0, -1.0]
NP = len(PILOT)
NS = NP + NL + MAXB
AP, AQ = 10, 16
MID = 2048
LF = 300

PATS = [
    [1.0] * 44 + [-1.0] * 44,
    [1.0] * 22 + [-1.0] * 44 + [1.0] * 22,
    [1.0] * 88,
]

H = None
KNOWN_LEN = None


def conv(a, b):
    if np is not None:
        return np.convolve(np.asarray(a, float), np.asarray(b, float))[:N]
    out = [0.0] * N
    for j, bj in enumerate(b):
        if bj == 0.0:
            continue
        for i in range(min(len(a), N - j)):
            out[i + j] += a[i] * bj
    return out


def shift(g, d):
    if np is not None:
        r = np.zeros(N)
        r[d:] = g[:N - d]
        return r
    return [0.0] * d + list(g[:N - d])


def lstsq(cols, y):
    if np is not None:
        return np.linalg.lstsq(np.column_stack(cols), y, rcond=None)[0]
    m = len(cols)
    a = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    for i in range(m):
        for j in range(i, m):
            a[i][j] = a[j][i] = sum(p * q for p, q in zip(cols[i], cols[j]))
        b[i] = sum(p * q for p, q in zip(cols[i], y))
        a[i][i] = a[i][i] * (1.0 + 1e-9) + 1e-12
    return gauss(a, b)


def symbols_for(msg):
    s = [0.0] * NS
    s[:NP] = PILOT
    v = len(msg) - 1
    for i in range(NL):
        s[NP + i] = 1.0 if (v >> (NL - 1 - i)) & 1 else -1.0
    for i, ch in enumerate(msg):
        s[NP + NL + i] = 1.0 if ch == '1' else -1.0
    return s


def predicted_peak(g, s):
    if np is not None:
        tot = np.zeros(N)
        for k, sk in enumerate(s):
            if sk:
                tot[k * T:] += sk * g[:N - k * T]
        return float(np.max(np.abs(tot)))
    tot = [0.0] * N
    for k, sk in enumerate(s):
        if sk:
            o = k * T
            for i in range(N - o):
                tot[o + i] += sk * g[i]
    return max(abs(v) for v in tot)


def baseline(h):
    global H
    H = None
    if len(h) < 100:
        return
    tail = sorted(h[-(len(h) // 5):])
    base = tail[len(tail) // 2]
    ht = [float(v) - base for v in h[:N]] + [0.0] * max(0, N - len(h))
    peak = max(abs(v) for v in ht)
    if peak < 3:
        return
    if min(h) >= 0 and max(h) <= 4095:
        limit = max(0.85 * min(base, 4095 - base), peak)
    else:
        limit = 3.0 * peak
    seed, prs = 12345, []
    for _ in range(NS):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        prs.append(1.0 if (seed >> 16) & 1 else -1.0)
    best = None
    for ci, pat in enumerate(PATS):
        g = conv(ht, pat)
        if np is not None:
            g = g / 2047.0
            en = float(np.dot(g, g))
        else:
            g = [v / 2047.0 for v in g]
            en = sum(v * v for v in g)
        amp = min(2047.0, limit / max(predicted_peak(g, prs), 1e-9))
        score = amp * amp * en
        if best is None or score > best[0]:
            best = (score, ci, g)
    step, acc = [], 0.0
    for v in ht:
        acc += v
        step.append(acc)
    if np is not None:
        ht, step = np.array(ht), np.array(step)
    H = {'ht': ht, 'cand': best[1], 'g': best[2], 'limit': limit, 'step': step}


def encode(msg):
    global KNOWN_LEN
    msg = ''.join(c for c in msg if c in '01')[:MAXB] or '0'
    KNOWN_LEN = len(msg)
    s = symbols_for(msg)
    if H is None:
        pat, amp = PATS[0], 1000.0
    else:
        pat = PATS[H['cand']]
        amp = min(2047.0, H['limit'] / max(predicted_peak(H['g'], s), 1e-9))
        amp = max(amp, 64.0)
    out = [MID] * N
    for k, sk in enumerate(s):
        if sk:
            o = k * T
            for i, p in enumerate(pat):
                out[o + i] = min(4095, max(0, int(round(MID + amp * sk * p))))
    return out


def sym_ls(heff, y, pat, use_step=True):
    g = conv(heff, pat)
    if np is not None:
        cols = [np.ones(N)]
        if use_step:
            cols.append(np.cumsum(np.asarray(heff, float))[:N])
        cols += [shift(g, k * T) for k in range(NS)]
        a = np.column_stack(cols)
        sc = np.sqrt((a * a).sum(axis=0)) + 1e-12
        w = np.linalg.lstsq(a / sc, y, rcond=None)[0]
        r = y - (a / sc) @ w
        return float(r @ r), list((w / sc)[-NS:])
    ne = 2 if use_step else 1
    m = ne + NS
    gm = [[0.0] * m for _ in range(m)]
    rhs = [0.0] * m
    ext = [[1.0] * N]
    if use_step:
        st, acc = [], 0.0
        for v in list(heff)[:N] + [0.0] * (N - len(heff)):
            acc += v
            st.append(acc)
        ext.append(st)
    for i in range(ne):
        for j in range(i, ne):
            gm[i][j] = gm[j][i] = sum(p * q for p, q in zip(ext[i], ext[j]))
        rhs[i] = sum(p * q for p, q in zip(ext[i], y))
        for k in range(NS):
            v = sum(p * q for p, q in zip(ext[i][k * T:], g))
            gm[i][ne + k] = gm[ne + k][i] = v
    for k in range(NS):
        rhs[ne + k] = sum(p * q for p, q in zip(y[k * T:], g))
    for lag in range(NS):
        pre, acc = [0.0], 0.0
        for p, q in zip(g[lag * T:], g):
            acc += p * q
            pre.append(acc)
        for j in range(lag, NS):
            n = N - j * T
            v = pre[min(n, len(pre) - 1)]
            gm[ne + j - lag][ne + j] = gm[ne + j][ne + j - lag] = v
    for i in range(m):
        gm[i][i] = gm[i][i] * (1.0 + 1e-9) + 1e-12
    w = gauss(gm, rhs)
    res = sum(v * v for v in y) - sum(p * q for p, q in zip(w, rhs))
    return res, w[-NS:]


def gauss(a, b):
    m = len(b)
    a = [row[:] + [b[i]] for i, row in enumerate(a)]
    for i in range(m):
        p = max(range(i, m), key=lambda r: abs(a[r][i]))
        a[i], a[p] = a[p], a[i]
        d = a[i][i]
        if abs(d) < 1e-300:
            continue
        ai = a[i]
        for r in range(i + 1, m):
            f = a[r][i] / d
            if f:
                ar = a[r]
                for c in range(i, m + 1):
                    ar[c] -= f * ai[c]
    w = [0.0] * m
    for i in range(m - 1, -1, -1):
        v = a[i][m] - sum(a[i][c] * w[c] for c in range(i + 1, m))
        w[i] = v / a[i][i] if abs(a[i][i]) > 1e-300 else 0.0
    return w


def ar_filter(v, ar):
    out = [0.0] * len(v)
    for t in range(len(v)):
        acc = v[t]
        for i in range(1, min(AP, t) + 1):
            acc += ar[i - 1] * out[t - i]
        if acc != acc or abs(acc) > 1e15:
            return None
        out[t] = acc
    return out


def arx_fit(x, y, t1):
    t0 = max(AP, AQ)
    if t1 - t0 < 3 * (AP + AQ):
        return []
    yl, xl, ol = list(y)[:t1], list(x)[:t1], [1.0] * t1
    y0 = yl[0]
    yl = [v - y0 for v in yl]
    res = []
    yf, xf, of = yl, xl, ol
    for _ in range(4):
        cols = [yf[t0 - i:t1 - i] for i in range(1, AP + 1)]
        cols += [xf[t0 - j:t1 - j] for j in range(AQ)]
        cols.append(of[t0:t1])
        tgt = yf[t0:t1]
        if np is not None:
            cols = [np.array(c, float) for c in cols]
            tgt = np.array(tgt, float)
        try:
            w = [float(v) for v in lstsq(cols, tgt)]
        except Exception:
            break
        ar, bq = w[:AP], w[AP:AP + AQ]
        h = ar_filter(bq + [0.0] * (N - AQ), ar)
        if h is None:
            break
        head = max(abs(v) for v in h[:N // 2])
        if head == 0.0 or max(abs(v) for v in h[N // 2:]) > 3.0 * head:
            break
        res.append(np.array(h) if np is not None else h)
        yf, xf, of = ar_filter(yl, ar), ar_filter(xl, ar), ar_filter(ol, ar)
        if yf is None or xf is None or of is None:
            break
    return res[:1] + res[-1:] if len(res) > 1 else res


def fir_fit(x, y):
    cols = [np.ones(N)]
    if H:
        cols += [H['step'], np.convolve(H['ht'], x)[:N]]
    nb = len(cols)
    a = np.column_stack(cols + [shift(x, d) for d in range(LF)])
    m = a.T @ a
    d = np.diag(m).copy()
    lam = np.zeros(len(d))
    lam[:nb] = 1e-9 * d[:nb]
    lam[nb:] = 1e-3 * d[nb:].mean()
    w = np.linalg.solve(m + np.diag(lam), a.T @ y)
    heff = np.zeros(N)
    heff[:LF] = w[nb:]
    if H:
        heff = heff + w[nb - 1] * H['ht']
    return heff


def build_x(s, pat):
    x = [0.0] * N
    for k, sk in enumerate(s):
        if sk:
            for i, p in enumerate(pat):
                x[k * T + i] = sk * p
    return np.array(x) if np is not None else x


def read_bits(c):
    if KNOWN_LEN is not None:
        ln = KNOWN_LEN
    else:
        ln = 0
        for i in range(NL):
            ln = ln * 2 + (1 if c[NP + i] > 0 else 0)
        ln = min(MAXB, ln + 1)
    return ''.join('1' if c[NP + NL + i] > 0 else '0' for i in range(ln))


def decode(y):
    y = [float(v) for v in y[:N]] + [0.0] * max(0, N - len(y))
    pat = PATS[H['cand']] if H else PATS[0]
    if np is not None:
        y = np.array(y)
    best = None

    def consider(heff, use_step=True):
        nonlocal best
        if heff is None:
            return
        try:
            res, c = sym_ls(heff, y, pat, use_step)
        except Exception:
            return
        if res == res and (best is None or res < best[0]):
            best = (res, c)

    if H:
        consider(H['ht'])
    pil = [PILOT[k] if k < NP else 0.0 for k in range(NS)]
    for hm in arx_fit(build_x(pil, pat), y, NP * T):
        consider(hm)
    if best is None:
        res, c = sym_ls([1.0], y, pat, False)
        if sum(p * q for p, q in zip(PILOT, c)) < 0:
            c = [-v for v in c]
        best = (res, c)
    prev = None
    for _ in range(3):
        bits = read_bits(best[1])
        if bits == prev:
            break
        prev = bits
        x = build_x(symbols_for(bits), pat)
        for hm in arx_fit(x, y, N):
            consider(hm)
        if np is not None:
            try:
                consider(fir_fit(x, y))
            except Exception:
                pass
    return read_bits(best[1])


def main():
    data = sys.stdin
    out = sys.stdout

    def next_line():
        while True:
            line = data.readline()
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
        else:
            vals = []
            for tok in payload.split():
                try:
                    vals.append(int(float(tok)))
                except ValueError:
                    pass
            if mode == '3':
                baseline(vals)
            else:
                out.write(decode(vals) + '\n')
                out.flush()


if __name__ == '__main__':
    main()
