import os
import sys
import time

try:
    import numpy as np
except Exception:
    np = None

START = time.time()
LAG_BUDGET = 3.5
TOTAL_BUDGET = 5.5


def read_lines():
    data = sys.stdin.buffer.read()
    from_file = False
    if not data.strip() and os.path.exists('input.txt'):
        with open('input.txt', 'rb') as f:
            data = f.read()
        from_file = True
    return [ln for ln in data.split(b'\n') if ln.strip()], from_file


def parse_series(line):
    text = line.decode('ascii', 'ignore')
    if np is None:
        try:
            return [float(v) for v in text.split()]
        except ValueError:
            return [float(v) for v in text.replace(',', ' ').replace(';', ' ').split()]
    try:
        arr = np.array(text.split(), dtype=np.float64)
    except ValueError:
        arr = np.array(text.replace(',', ' ').replace(';', ' ').split(), dtype=np.float64)
    return np.nan_to_num(arr)


def find_lags(y, refs, n):
    m = len(y)
    k = len(refs)
    lags = np.zeros(k, dtype=np.int64)
    peaks = np.zeros(k)
    nonzero = np.abs(y[y != 0])
    if nonzero.size == 0:
        return lags
    limit = 3.0 * float(np.median(nonzero))
    yc = np.clip(y, -limit, limit)
    longest = max(len(r) for r in refs)
    size = 1 << (m + longest).bit_length()
    spectrum = np.fft.rfft(yc, size)
    cy = np.concatenate(([0.0], np.cumsum(yc * yc)))
    for i, r in enumerate(refs):
        if time.time() - START > LAG_BUDGET:
            break
        length = len(r)
        if length == 0:
            continue
        cr = np.concatenate(([0.0], np.cumsum(r * r)))
        corr = np.fft.irfft(spectrum * np.conj(np.fft.rfft(r, size)), size)
        shift = np.arange(-(length - 1), m)
        lo = np.maximum(shift, 0)
        hi = np.minimum(shift + length, m)
        energy = (cy[hi] - cy[lo]) * (cr[hi - shift] - cr[lo - shift])
        valid = (hi - lo) >= max(1, min(length, m) // 2)
        floor = 1e-12 * cy[-1] * cr[-1] + 1e-300
        score = np.where(valid, np.abs(corr[shift % size]) / np.sqrt(energy + floor), 0.0)
        best = int(np.argmax(score))
        lags[i] = shift[best]
        peaks[i] = score[best]
    if all(len(r) == m for r in refs):
        top = np.argsort(-peaks)[:n]
        if 2 * int(np.sum(lags[top] == 0)) >= len(top):
            lags[:] = 0
    return lags


def place(r, lag, m):
    out = np.zeros(m)
    lo = max(0, lag)
    hi = min(m, lag + len(r))
    if hi > lo:
        out[lo:hi] = r[lo - lag:hi - lag]
    return out


def windows(x, w):
    count = -(-len(x) // w)
    padded = np.zeros(count * w)
    padded[:len(x)] = x
    return padded.reshape(count, w)


def score(residual_w, ref_w):
    er = np.einsum('ij,ij->i', ref_w, ref_w)
    total = float(er.sum())
    if total <= 0.0:
        return 0.0
    ey = np.einsum('ij,ij->i', residual_w, residual_w)
    num = np.einsum('ij,ij->i', residual_w, ref_w)
    rho = num / np.sqrt(er * ey + 1e-300)
    return abs(float(np.dot(er, rho))) / total


def residual(y_w, chosen_w):
    a = np.stack(chosen_w)
    gram = np.einsum('inw,jnw->nij', a, a)
    rhs = np.einsum('inw,nw->ni', a, y_w)
    s = a.shape[0]
    ridge = 1e-6 * np.trace(gram, axis1=1, axis2=2) / s + 1e-12
    gram = gram + ridge[:, None, None] * np.eye(s)
    gains = np.linalg.solve(gram, rhs[:, :, None])[:, :, 0]
    return y_w - np.einsum('ni,inw->nw', gains, a)


def solve_numpy(n, y, refs):
    k = len(refs)
    m = len(y)
    lags = find_lags(y, refs, n)
    w = min(m, max(16, 8 * n, min(1024, m // 50)))
    y_w = windows(y, w)
    res_w = y_w
    chosen = []
    chosen_w = []
    while len(chosen) < n:
        scores = np.full(k, -1.0)
        for i in range(k):
            if i not in chosen:
                scores[i] = score(res_w, windows(place(refs[i], int(lags[i]), m), w))
        if time.time() - START > TOTAL_BUDGET:
            chosen.extend(int(i) for i in np.argsort(-scores)[:n - len(chosen)])
            break
        best = int(np.argmax(scores))
        chosen.append(best)
        chosen_w.append(windows(place(refs[best], int(lags[best]), m), w))
        res_w = residual(y_w, chosen_w)
    return sorted(chosen)


def solve_plain(n, y, refs):
    m = len(y)
    w = min(m, max(16, min(1024, m // 50)))
    scores = []
    for i, r in enumerate(refs):
        length = min(m, len(r))
        total = 0.0
        acc = 0.0
        for start in range(0, length, w):
            ys = y[start:start + w]
            rs = r[start:start + w]
            er = sum(v * v for v in rs)
            ey = sum(v * v for v in ys)
            if er > 0.0 and ey > 0.0:
                acc += er * sum(p * q for p, q in zip(ys, rs)) / (er * ey) ** 0.5
            total += er
        scores.append((abs(acc) / total if total > 0.0 else 0.0, i))
    scores.sort(reverse=True)
    return sorted(i for _, i in scores[:n])


def main():
    lines, from_file = read_lines()
    if not lines:
        return
    head = lines[0].split()
    n = int(float(head[0]))
    if len(head) >= 2:
        k = int(float(head[1]))
        body = lines[1:]
    else:
        k = int(float(lines[1].split()[0]))
        body = lines[2:]
    if not body:
        return
    y = parse_series(body[0])
    refs = [parse_series(ln) for ln in body[1:k + 1]]
    k = len(refs)
    n = max(0, min(n, k))
    if n == k or len(y) == 0:
        answer = list(range(k))[:n]
    elif np is not None:
        answer = solve_numpy(n, y, refs)
    else:
        answer = solve_plain(n, y, refs)
    text = ' '.join(map(str, answer))
    sys.stdout.write(text + '\n')
    if from_file:
        with open('output.txt', 'w') as f:
            f.write(text + '\n')


if __name__ == '__main__':
    main()
