#!/usr/bin/env python3
"""IR-bench SHA-256 proof-of-work.

Bitcoin-style double SHA-256:
    digest = SHA256(SHA256(header_76_bytes || nonce_4_bytes))

Difficulty 20 means the final 32-byte digest must contain at least
20 leading zero bits.

Timing model:
    Every optical exposure is assumed to take 1 ms.
    Reported bench hash rate = validated hashes / (exposures * 0.001 s).

The optical bench performs lane-parallel nonce arithmetic and simulated
camera exposures. The CPU computes and validates every SHA-256 hash.
The optical system does not compute SHA-256.

Run:
    python -m pip install pillow
    python sha256_pow_ir_bench.py
"""

import csv
import hashlib
import math
import random
import struct
import time
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

from PIL import Image, ImageDraw, ImageTk

DEFAULT_CELLS = 64
MAX_CELLS = 1024
DEFAULT_DIFFICULTY = 10
MAX_DIFFICULTY = 32
MAX_HASHES = 20_000_000

EXPOSURE_TIME_MS = 0.0002
EXPOSURE_TIME_S = EXPOSURE_TIME_MS / 1000.0

DPI = 2540
SIZE_MM = 20
PX_PER_MM = 100
DIAMETER_MM = 0.7
SHOT_K = 0.06


# ---------------------------------------------------------------------------
# Printable 20 mm mask
# ---------------------------------------------------------------------------

def grid_shape(cells):
    cols = min(cells, math.ceil(math.sqrt(2 * cells)))
    return cols, math.ceil(cells / cols)


def mm(value):
    return round(value * PX_PER_MM)


def make_mask(cells):
    if not 1 <= cells <= MAX_CELLS:
        raise ValueError(f'Cells must be 1..{MAX_CELLS}.')

    cols, rows = grid_shape(cells)
    pitch_x = 16 / cols
    pitch_y = 16 / rows
    diameter = min(DIAMETER_MM, pitch_x / 4.5, pitch_y / 4)
    dx = pitch_x * 0.25
    dy = pitch_y * 0.16

    image = Image.new('1', (mm(SIZE_MM), mm(SIZE_MM)), 0)
    draw = ImageDraw.Draw(image)
    layout = []
    radius = mm(diameter) / 2

    for i in range(cells):
        cx = 2 + (i % cols + 0.5) * pitch_x
        cy = 2 + (i // cols + 0.5) * pitch_y

        for label, ox, oy in [
            ('a', -dx, -dy),
            ('b', 0, -dy),
            ('carry', dx, -dy),
            ('ref_A', -dx / 2, dy),
            ('ref_B', dx / 2, dy),
        ]:
            x = mm(cx + ox)
            y = mm(cy + oy)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=1
            )
            layout.append((i + 1, label, cx + ox, cy + oy, diameter))

    for x, y in [(1, 1), (19, 1), (1, 19), (19, 19)]:
        x = mm(x)
        y = mm(y)
        draw.line(
            (x - mm(0.3), y, x + mm(0.3), y),
            fill=1,
            width=mm(0.06)
        )
        draw.line(
            (x, y - mm(0.3), x, y + mm(0.3)),
            fill=1,
            width=mm(0.06)
        )

    draw.rectangle((mm(1.6), mm(0.9), mm(1.8), mm(1.1)), fill=1)

    meta = {
        'cells': cells,
        'cols': cols,
        'rows': rows,
        'diameter': diameter,
        'pitch_x': pitch_x,
        'pitch_y': pitch_y,
    }
    return image, layout, meta


def export_mask(folder, cells):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    image, layout, meta = make_mask(cells)
    stem = f'physical_adder_mask_20mm_{cells}cells'
    path = folder / (stem + '.png')
    image.save(path, dpi=(DPI, DPI))

    with (folder / (stem + '.csv')).open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'cell',
            'aperture',
            'x_mm',
            'y_mm',
            'diameter_mm',
        ])
        writer.writerows(layout)

    (folder / (stem + '_notes.txt')).write_text(
        f'20 x 20 mm mask at {DPI} DPI.\n'
        f'{cells} cells: {meta["cols"]} columns x {meta["rows"]} rows.\n'
        f'Nominal aperture diameter: {meta["diameter"]:.6f} mm.\n'
        'White = open. Black = opaque. Print at actual size.\n'
        'This mask implements bench adder cells, not SHA-256 rounds.\n'
    )

    return path, image, meta


# ---------------------------------------------------------------------------
# Optical full-adder bench
# ---------------------------------------------------------------------------

class ITOPlate:
    """Simulated ITO IR-blocking plate. Parameter values are assumptions."""

    def __init__(
        self,
        enabled=False,
        extinction_db=20.0,
        loss=0.08,
        reject_db=15.0,
        tau_us=50.0,
        settle_us=250.0,
    ):
        self.enabled = bool(enabled)
        self.extinction_db = extinction_db
        self.loss = loss
        self.reject_db = reject_db
        self.tau_us = tau_us
        self.settle_us = settle_us

    @property
    def p_on(self):
        return 1.0 - self.loss

    @property
    def p_off(self):
        return 10 ** (-self.extinction_db / 10)

    @property
    def ambient_factor(self):
        return 10 ** (-self.reject_db / 10)

    @property
    def decay(self):
        if self.tau_us <= 0:
            return 0.0
        return math.exp(-self.settle_us / self.tau_us)

    def step(self, current, target_bit):
        target = self.p_on if target_bit else self.p_off
        return target + (current - target) * self.decay


class CameraBench:
    def __init__(
        self,
        cells,
        exposure=0.22,
        leak=0.001,
        noise=0.002,
        seed=1,
        led_a=True,
        led_b=True,
        ito=None,
        ambient=0.04,
        xtalk=0.02,
    ):
        self.cells = cells
        self.exposure = exposure
        self.leak = leak
        self.noise = noise
        self.rng = random.Random(seed)
        self.led_a = led_a
        self.led_b = led_b
        self.ito = ito if ito is not None else ITOPlate(False)
        self.ambient = ambient
        self.xtalk = xtalk

        self.itx = [
            [self.ito.p_off] * 5
            for _ in range(cells)
        ]
        self.pbits = [
            [0] * 5
            for _ in range(cells)
        ]

        self.exposures = 0
        self.last = None
        self.worst_margin = 0.0
        self.err_sq = 0.0
        self.err_n = 0
        self.plate_time_us = 0.0
        self.useful_slots = 0
        self.idle_slots = 0

    @property
    def rms_error(self):
        if self.err_n == 0:
            return 0.0
        return math.sqrt(self.err_sq / self.err_n)

    @property
    def exposure_seconds(self):
        return self.exposures * EXPOSURE_TIME_S

    def capture(self, inputs, active=None):
        n = self.cells

        if len(inputs) != n:
            raise ValueError(
                f'Exactly {n} input lanes are required.'
            )

        act = [True] * n if active is None else list(active)

        if len(act) != n:
            raise ValueError(
                f'Exactly {n} active flags are required.'
            )

        self.useful_slots += sum(act)
        self.idle_slots += n - sum(act)

        use_ito = self.ito.enabled
        ambient = self.ambient * (
            self.ito.ambient_factor if use_ito else 1.0
        )
        dark = 0.01 + ambient
        noise_sigma = math.sqrt(
            self.noise ** 2 + SHOT_K ** 2 * ambient
        )
        exposure = self.exposure
        lit = sum(act) if use_ito else n
        stray = self.xtalk * exposure * lit / n
        p_on = self.ito.p_on if use_ito else 1.0
        closed = self.leak * (
            self.ito.p_off if use_ito else 1.0
        )
        lambda_closed = closed / p_on
        led_a = float(self.led_a)
        led_b = float(self.led_b)

        def transmission(bit):
            return self.leak + (1.0 - self.leak) * bit

        def pixel(value):
            return max(
                0.0,
                min(
                    1.0,
                    value + self.rng.gauss(0, noise_sigma)
                )
            )

        rows = []
        changed = False

        for i, (a, b, carry) in enumerate(inputs):
            if a not in (0, 1) or b not in (0, 1) or carry not in (0, 1):
                raise ValueError(
                    'Optical inputs must be binary.'
                )

            if use_ito:
                idle = not act[i]
                target_bits = (
                    (0, 0, 0, 0, 0)
                    if idle
                    else (a, b, carry, 1, 1)
                )

                if list(target_bits) != self.pbits[i]:
                    changed = True
                    self.pbits[i] = list(target_bits)

                for j in range(5):
                    self.itx[i][j] = self.ito.step(
                        self.itx[i][j],
                        target_bits[j]
                    )

                pa, pb, pc, ref_a, ref_b = self.itx[i]

            else:
                pa = pb = pc = ref_a = ref_b = 1.0

            signal = dark + stray + exposure * (
                led_a * (
                    transmission(a) * pa
                    + transmission(b) * pb
                )
                + led_b * transmission(carry) * pc
            )
            black = dark
            ref_a_level = dark + stray + exposure * led_a * ref_a
            ref_b_level = dark + stray + exposure * led_b * ref_b
            baseline = dark + stray + exposure * closed * (
                2 * led_a + led_b
            )

            rows.append({
                'signal': pixel(signal),
                'black': pixel(black),
                'refa': pixel(ref_a_level),
                'refb': pixel(ref_b_level),
                'baseline': pixel(baseline),
            })

        self.exposures += 1

        if use_ito and changed:
            self.plate_time_us += self.ito.settle_us

        decoded = []

        for i, row in enumerate(rows):
            if use_ito and not act[i]:
                decoded.append({
                    'level': 0.0,
                    'k': 0,
                    'sum': 0,
                    'carry': 0,
                })
                continue

            ref_a = row['refa'] - row['black']
            ref_b = row['refb'] - row['black']

            if min(ref_a, ref_b) < 0.04:
                raise RuntimeError(
                    'Insufficient LED reference signal.'
                )

            if abs(ref_a - ref_b) > 0.25 * max(ref_a, ref_b):
                raise RuntimeError(
                    'Unequal LED references.'
                )

            level = (
                row['signal'] - row['baseline']
            ) / (
                (ref_a + ref_b) / 2 * (1 - lambda_closed)
            )
            k = round(level)

            self.worst_margin = max(
                self.worst_margin,
                abs(level - k)
            )
            self.err_sq += (level - k) ** 2
            self.err_n += 1

            if row['signal'] > 0.98:
                raise RuntimeError(
                    'Camera signal saturated. Lower exposure.'
                )

            if not 0 <= k <= 3 or abs(level - k) > 0.35:
                raise RuntimeError(
                    'Ambiguous camera level. '
                    'Lower exposure or noise.'
                )

            decoded.append({
                'level': level,
                'k': k,
                'sum': k % 2,
                'carry': int(k >= 2),
            })

        self.last = {
            'inputs': list(inputs),
            'rows': rows,
            'decoded': decoded,
        }
        return decoded


def optical_add(
    bench,
    x_values,
    y_values,
    width,
    active=None,
):
    """Lane-parallel optical addition."""
    n = bench.cells

    if len(x_values) != n or len(y_values) != n:
        raise ValueError(
            f'optical_add requires exactly {n} lanes.'
        )

    carry = [0] * n
    result = [0] * n
    act = [True] * n if active is None else list(active)

    for bit in range(width):
        inputs = []

        for i in range(n):
            inputs.append((
                (x_values[i] >> bit) & 1,
                (y_values[i] >> bit) & 1,
                carry[i],
            ))

        decoded = bench.capture(inputs, act)

        for i, d in enumerate(decoded):
            result[i] |= d['sum'] << bit
            carry[i] = d['carry']

        yield {
            'bit': bit,
            'width': width,
            'stage': (
                f'optical add bit {bit + 1}/{width}'
            ),
        }

    if any(carry):
        raise RuntimeError('Optical addition overflow.')

    return result


# ---------------------------------------------------------------------------
# SHA-256 proof-of-work
# ---------------------------------------------------------------------------

def sha256d(data):
    return hashlib.sha256(
        hashlib.sha256(data).digest()
    ).digest()


def leading_zero_bits(digest):
    count = 0

    for byte in digest:
        if byte == 0:
            count += 8
        else:
            count += 8 - byte.bit_length()
            break

    return count


def default_header():
    version = struct.pack('<I', 0x20000000)
    previous = bytes(32)
    merkle = hashlib.sha256(
        b'IR-bench example merkle root'
    ).digest()
    timestamp = struct.pack('<I', int(time.time()))
    bits = struct.pack('<I', 0x1702b3f7)

    return version + previous + merkle + timestamp + bits


def parse_header_hex(text):
    try:
        raw = bytes.fromhex(''.join(text.split()))
    except ValueError:
        raise ValueError(
            'Header must be valid hexadecimal.'
        )

    if len(raw) != 76:
        raise ValueError(
            f'Header must be exactly 76 bytes, not {len(raw)}.'
        )

    return raw


def pow_job(
    bench,
    header,
    start_nonce,
    end_nonce,
    difficulty,
    report,
):
    """Optical nonce generation plus real CPU SHA-256 validation."""
    n = bench.cells
    total = end_nonce - start_nonce + 1
    expected = 2 ** difficulty

    if start_nonce < 0 or end_nonce > 0xFFFFFFFF:
        raise ValueError(
            'Nonce values must fit in 32 bits.'
        )

    if end_nonce < start_nonce:
        raise ValueError(
            'Nonce end must be greater than or equal to nonce start.'
        )

    if total > MAX_HASHES:
        raise ValueError(
            f'Nonce range too large: '
            f'{total:,} > {MAX_HASHES:,}.'
        )

    if expected > MAX_HASHES:
        raise ValueError(
            f'Difficulty too high: expected about '
            f'{expected:,} hashes.'
        )

    report(
        f'PoW started: {n} lanes, difficulty {difficulty}, '
        f'nonce range {start_nonce:,}..{end_nonce:,}, '
        f'{total:,} candidates, expected about '
        f'{expected:,} hashes.'
    )
    report(
        f'Timing model: {EXPOSURE_TIME_MS:.1f} ms per optical exposure. '
        'Bench hash rate = hashes / (exposures * 0.001 s).'
    )
    report(
        'CPU computes SHA256(SHA256(header||nonce)); '
        'optics performs lane addition and exposures.'
    )

    started = time.perf_counter()
    hashes = 0
    batches = 0
    solution = None
    nonce = start_nonce

    while nonce <= end_nonce and solution is None:
        lane_count = min(n, end_nonce - nonce + 1)
        active = [i < lane_count for i in range(n)]
        base = [nonce] * n
        offsets = list(range(n))

        adder = optical_add(
            bench,
            base,
            offsets,
            33,
            active,
        )

        while True:
            try:
                event = next(adder)
            except StopIteration as stop:
                optical_nonces = stop.value
                break

            exposure_hash_rate = hashes / max(
                bench.exposure_seconds,
                EXPOSURE_TIME_S
            )

            event['stage'] = (
                f'batch {batches + 1}, '
                f'lanes {lane_count}, '
                f'base nonce {nonce:,}, '
                f'{exposure_hash_rate:,.0f} H/s at '
                f'{EXPOSURE_TIME_MS:.1f} ms/exposure'
            )
            yield event

        for lane in range(lane_count):
            optical_nonce = optical_nonces[lane]
            expected_nonce = nonce + lane

            if optical_nonce != expected_nonce:
                raise RuntimeError(
                    f'Optical nonce mismatch: lane {lane} gave '
                    f'{optical_nonce}, expected {expected_nonce}.'
                )

            block_header = header + struct.pack(
                '>I',
                expected_nonce
            )
            digest = sha256d(block_header)
            hashes += 1
            zeros = leading_zero_bits(digest)

            if zeros >= difficulty:
                solution = (
                    expected_nonce,
                    digest,
                    zeros,
                )
                break

        batches += 1
        nonce += n

        exposure_hash_rate = hashes / max(
            bench.exposure_seconds,
            EXPOSURE_TIME_S
        )

        yield {
            'bit': 0,
            'width': 1,
            'stage': (
                f'batch {batches}, '
                f'{hashes:,} hashes, '
                f'{exposure_hash_rate:,.0f} H/s at '
                f'{EXPOSURE_TIME_MS:.1f} ms/exposure, '
                f'{bench.exposures:,} exposures'
            ),
        }

    elapsed = max(
        time.perf_counter() - started,
        1e-9
    )
    cpu_hash_rate = hashes / elapsed
    exposure_seconds = bench.exposure_seconds
    exposure_hash_rate = (
        hashes / exposure_seconds
        if exposure_seconds > 0
        else 0.0
    )
    exposures_per_hash = bench.exposures / max(1, hashes)

    if solution is not None:
        found_nonce, digest, zeros = solution
        check_header = header + struct.pack(
            '>I',
            found_nonce
        )
        check_digest = sha256d(check_header)

        if check_digest != digest:
            raise RuntimeError(
                'Final SHA-256 validation failed.'
            )

        if leading_zero_bits(check_digest) < difficulty:
            raise RuntimeError(
                'Final difficulty validation failed.'
            )

        report(f'VALID SOLUTION: nonce {found_nonce}')
        report(f'Leading zero bits: {zeros}')
        report(f'Digest: {digest.hex()}')

        return {
            'found': True,
            'nonce': found_nonce,
            'digest': digest.hex(),
            'zeros': zeros,
            'hashes': hashes,
            'seconds': elapsed,
            'cpu_hash_rate': cpu_hash_rate,
            'exposure_seconds': exposure_seconds,
            'exposure_hash_rate': exposure_hash_rate,
            'exposures': bench.exposures,
            'batches': batches,
            'exposures_per_hash': exposures_per_hash,
        }

    report(
        'No valid nonce found in the requested range.'
    )

    return {
        'found': False,
        'nonce': None,
        'digest': None,
        'zeros': 0,
        'hashes': hashes,
        'seconds': elapsed,
        'cpu_hash_rate': cpu_hash_rate,
        'exposure_seconds': exposure_seconds,
        'exposure_hash_rate': exposure_hash_rate,
        'exposures': bench.exposures,
        'batches': batches,
        'exposures_per_hash': exposures_per_hash,
    }


def cpu_pow_test(difficulty=12, max_nonce=1_000_000):
    """Fast CPU-only diagnostic for hashing and difficulty checking."""
    header = default_header()
    started = time.perf_counter()

    for nonce in range(max_nonce):
        digest = sha256d(
            header + struct.pack('>I', nonce)
        )

        if leading_zero_bits(digest) >= difficulty:
            elapsed = time.perf_counter() - started

            return {
                'found': True,
                'nonce': nonce,
                'digest': digest.hex(),
                'zeros': leading_zero_bits(digest),
                'hashes': nonce + 1,
                'seconds': elapsed,
                'hash_rate': (nonce + 1) / elapsed,
            }

    elapsed = time.perf_counter() - started

    return {
        'found': False,
        'nonce': None,
        'digest': None,
        'zeros': 0,
        'hashes': max_nonce,
        'seconds': elapsed,
        'hash_rate': max_nonce / elapsed,
    }


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('IR bench SHA-256 PoW')
        self.geometry('1280x850')

        self.job = None
        self.mode = None
        self.bench = None
        self.header = default_header()
        self.difficulty = DEFAULT_DIFFICULTY
        self.nonce_start = 0
        self.nonce_end = 1_048_575
        self.x_value = 0
        self.y_value = 0
        self.delay = 0

        self.last_camera = None
        self.display_cells = DEFAULT_CELLS
        self.cell_labels = []
        self.mask_window = None

        self.log_lines = []
        self.status = tk.StringVar(
            value='Ready. Press “Run SHA-256 PoW”. '
                  'Errors appear in the log.'
        )

        self.led_a = tk.BooleanVar(value=True)
        self.led_b = tk.BooleanVar(value=True)
        self.ito_enabled = tk.BooleanVar(value=True)

        self._build_controls()
        self._build_canvas()
        self._build_log()

        self.say(
            'SHA-256 PoW: CPU validates every hash; '
            'optical bench performs nonce arithmetic.'
        )
        self.say(
            f'Timing model: {EXPOSURE_TIME_MS:.1f} ms per exposure. '
            'Hash rate is calculated from exposure count.'
        )
        self.say(
            'Default: 64 lanes, difficulty 20, '
            'nonce 0..1,048,575.'
        )
        self.draw()

    def _build_controls(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill='x')

        self.vars = {
            'cells': tk.StringVar(value=str(DEFAULT_CELLS)),
            'difficulty': tk.StringVar(
                value=str(DEFAULT_DIFFICULTY)
            ),
            'nonce_start': tk.StringVar(value='0'),
            'nonce_end': tk.StringVar(value='1048575'),
            'header': tk.StringVar(
                value=default_header().hex()
            ),
            'exposure': tk.StringVar(value='0.22'),
            'leak': tk.StringVar(value='0.001'),
            'noise': tk.StringVar(value='0.002'),
            'seed': tk.StringVar(value='1'),
            'delay': tk.StringVar(value='0'),
            'ambient': tk.StringVar(value='0.04'),
            'ito_er': tk.StringVar(value='20'),
            'ito_loss': tk.StringVar(value='8'),
            'ito_rej': tk.StringVar(value='15'),
            'ito_tau': tk.StringVar(value='50'),
            'ito_settle': tk.StringVar(value='250'),
        }

        # Live update whenever the Cells field changes.
        self.vars['cells'].trace_add(
            'write',
            self.on_cells_changed
        )

        fields = [
            ('cells', 'Cells (1–1024)'),
            ('difficulty', 'Difficulty zero bits'),
            ('nonce_start', 'Nonce start'),
            ('nonce_end', 'Nonce end'),
            ('exposure', 'Exposure'),
            ('leak', 'Leak'),
            ('noise', 'Noise sigma'),
            ('seed', 'Seed'),
            ('delay', 'GUI delay ms'),
        ]

        self.entries = []

        for column, (key, label) in enumerate(fields):
            ttk.Label(
                top,
                text=label
            ).grid(
                row=0,
                column=column,
                padx=4,
                sticky='w'
            )

            entry = ttk.Entry(
                top,
                textvariable=self.vars[key],
                width=13
            )
            entry.grid(row=1, column=column, padx=4)
            self.entries.append(entry)

        header_frame = ttk.Frame(self, padding=(8, 0))
        header_frame.pack(fill='x')

        ttk.Label(
            header_frame,
            text=(
                '76-byte block-header hex: version | previous hash | '
                'merkle root | time | bits'
            )
        ).pack(anchor='w')

        header_entry = ttk.Entry(
            header_frame,
            textvariable=self.vars['header']
        )
        header_entry.pack(fill='x')
        self.entries.append(header_entry)

        ito_frame = ttk.Frame(self, padding=(8, 4))
        ito_frame.pack(fill='x')

        ttk.Checkbutton(
            ito_frame,
            text='ITO plate (simulated)',
            variable=self.ito_enabled
        ).pack(side='left', padx=4)

        ito_fields = [
            ('ambient', 'Ambient IR'),
            ('ito_er', 'Block ER dB'),
            ('ito_loss', 'Open loss %'),
            ('ito_rej', 'Ambient reject dB'),
            ('ito_tau', 'Plate tau us'),
            ('ito_settle', 'Settle us'),
        ]

        for key, label in ito_fields:
            ttk.Label(
                ito_frame,
                text=label
            ).pack(side='left', padx=(8, 2))

            entry = ttk.Entry(
                ito_frame,
                textvariable=self.vars[key],
                width=8
            )
            entry.pack(side='left')
            self.entries.append(entry)

        button_bar = ttk.Frame(self, padding=8)
        button_bar.pack(fill='x')

        self.buttons = []

        for label, command in [
            ('Run SHA-256 PoW', lambda: self.start('pow')),
            ('Full-adder test', lambda: self.start('truth')),
            ('Optical X + Y', lambda: self.start('add')),
            ('CPU PoW test', self.run_cpu_test),
            ('Export/show mask', self.show_mask),
            ('Stop', self.stop),
        ]:
            button = ttk.Button(
                button_bar,
                text=label,
                command=command
            )
            button.pack(side='left', padx=4)
            self.buttons.append(button)

        ttk.Checkbutton(
            button_bar,
            text='LED A',
            variable=self.led_a
        ).pack(side='left', padx=8)

        ttk.Checkbutton(
            button_bar,
            text='LED B',
            variable=self.led_b
        ).pack(side='left')

        ttk.Label(
            self,
            textvariable=self.status,
            padding=8,
            wraplength=1240
        ).pack(fill='x')

    def _build_canvas(self):
        frame = ttk.Frame(self)
        frame.pack(fill='both', expand=True, padx=8)

        self.canvas = tk.Canvas(
            frame,
            bg='#101827',
            highlightthickness=0
        )

        scrollbar = ttk.Scrollbar(
            frame,
            orient='vertical',
            command=self.canvas.yview
        )

        self.canvas.configure(
            yscrollcommand=scrollbar.set
        )
        scrollbar.pack(side='right', fill='y')
        self.canvas.pack(
            side='left',
            fill='both',
            expand=True
        )
        self.canvas.bind(
            '<Configure>',
            lambda event: self.draw()
        )

    def _build_log(self):
        self.log = tk.Text(
            self,
            height=10,
            wrap='word',
            font=('TkFixedFont', 10)
        )
        self.log.pack(fill='x', padx=8, pady=8)
        self.log.configure(state='disabled')

        ttk.Label(
            self,
            text=(
                'Optical bench: lane-parallel nonce addition and camera '
                'exposures. CPU: double SHA-256, difficulty target, and '
                'final validation. No all-optical SHA-256 claim.'
            ),
            padding=8
        ).pack(fill='x')

    def say(self, text):
        self.log_lines.append(text)

        self.log.configure(state='normal')
        self.log.delete('1.0', 'end')
        self.log.insert(
            'end',
            '\n'.join(self.log_lines)
        )
        self.log.see('end')
        self.log.configure(state='disabled')

    def show_exception(self, title, exc):
        text = ''.join(
            traceback.format_exception(
                type(exc),
                exc,
                exc.__traceback__
            )
        )

        self.say(f'{title}: {exc}')
        self.say(text)
        self.status.set(f'{title}: {exc}')

    def on_cells_changed(self, *args):
        try:
            cells = int(self.vars['cells'].get())
        except ValueError:
            return

        if not 1 <= cells <= MAX_CELLS:
            return

        self.display_cells = cells
        self.cell_labels = []
        self.last_camera = None

        self.status.set(
            f'Cells set to {cells}. '
            f'Next PoW run will use {cells} parallel nonce lanes.'
        )
        self.draw()

        if (
            self.mask_window is not None
            and self.mask_window.winfo_exists()
        ):
            try:
                image, layout, meta = make_mask(cells)
                self.update_mask_window(
                    cells,
                    image,
                    meta
                )
            except Exception as exc:
                self.show_exception(
                    'Mask preview failed',
                    exc
                )

    def update_mask_window(self, cells, image, meta):
        if (
            self.mask_window is None
            or not self.mask_window.winfo_exists()
        ):
            return

        for child in self.mask_window.winfo_children():
            child.destroy()

        try:
            resample = Image.Resampling.NEAREST
        except AttributeError:
            resample = Image.NEAREST

        preview = image.convert('RGB').resize(
            (600, 600),
            resample
        )
        photo = ImageTk.PhotoImage(preview)

        label = ttk.Label(
            self.mask_window,
            image=photo
        )
        label.image = photo
        label.pack(padx=8, pady=8)

        ttk.Label(
            self.mask_window,
            text=(
                f'{cells} cells | '
                f'{meta["cols"]} x {meta["rows"]} | '
                f'aperture {meta["diameter"]:.3f} mm\n'
                'Preview only. Use Export/show mask for PNG.'
            ),
            padding=8
        ).pack()

    def cell_count(self):
        try:
            value = int(self.vars['cells'].get())
        except ValueError:
            raise ValueError('Cells must be an integer.')

        if not 1 <= value <= MAX_CELLS:
            raise ValueError(
                f'Cells must be 1..{MAX_CELLS}.'
            )

        return value

    def make_bench(self):
        cells = self.cell_count()

        exposure = float(self.vars['exposure'].get())
        leak = float(self.vars['leak'].get())
        noise = float(self.vars['noise'].get())
        seed = int(self.vars['seed'].get())
        ambient = float(self.vars['ambient'].get())

        if not 0 <= exposure <= 1:
            raise ValueError('Exposure must be 0..1.')

        if not 0 <= leak < 0.5:
            raise ValueError('Leak must be 0..<0.5.')

        if not 0 <= noise <= 1:
            raise ValueError(
                'Noise sigma must be 0..1.'
            )

        if not 0 <= ambient <= 0.5:
            raise ValueError(
                'Ambient IR must be 0..0.5.'
            )

        ito = ITOPlate(
            enabled=self.ito_enabled.get(),
            extinction_db=float(self.vars['ito_er'].get()),
            loss=float(self.vars['ito_loss'].get()) / 100,
            reject_db=float(self.vars['ito_rej'].get()),
            tau_us=float(self.vars['ito_tau'].get()),
            settle_us=float(self.vars['ito_settle'].get()),
        )

        return CameraBench(
            cells=cells,
            exposure=exposure,
            leak=leak,
            noise=noise,
            seed=seed,
            led_a=self.led_a.get(),
            led_b=self.led_b.get(),
            ito=ito,
            ambient=ambient,
        )

    def start(self, mode):
        if self.job is not None:
            return

        try:
            self.mode = mode
            self.bench = self.make_bench()
            self.display_cells = self.bench.cells
            self.last_camera = None
            self.cell_labels = []
            self.delay = int(self.vars['delay'].get())

            if mode == 'pow':
                self.difficulty = int(
                    self.vars['difficulty'].get()
                )
                self.nonce_start = int(
                    self.vars['nonce_start'].get()
                )
                self.nonce_end = int(
                    self.vars['nonce_end'].get()
                )
                self.header = parse_header_hex(
                    self.vars['header'].get()
                )

                if not 1 <= self.difficulty <= MAX_DIFFICULTY:
                    raise ValueError(
                        f'Difficulty must be '
                        f'1..{MAX_DIFFICULTY}.'
                    )

                self.job = pow_job(
                    self.bench,
                    self.header,
                    self.nonce_start,
                    self.nonce_end,
                    self.difficulty,
                    self.say,
                )

            elif mode == 'truth':
                self.job = self.truth_job()

            elif mode == 'add':
                self.x_value = int(
                    self.vars['nonce_start'].get()
                )
                self.y_value = int(
                    self.vars['nonce_end'].get()
                )

                if self.x_value < 0 or self.y_value < 0:
                    raise ValueError(
                        'X and Y must be non-negative.'
                    )

                width = (
                    self.x_value + self.y_value
                ).bit_length() + 1

                self.job = optical_add(
                    self.bench,
                    [self.x_value] * self.bench.cells,
                    [self.y_value] * self.bench.cells,
                    width,
                )

            self.set_ui_running(True)
            self.status.set(f'{mode} running...')
            self.after(1, self.tick)

        except Exception as exc:
            self.job = None
            self.set_ui_running(False)
            self.show_exception(
                'Could not start',
                exc
            )

    def truth_job(self):
        n = self.bench.cells
        tested = 0

        for start in range(0, 8, n):
            ids = list(range(start, min(8, start + n)))
            inputs = []

            for value in ids:
                inputs.append((
                    (value >> 2) & 1,
                    (value >> 1) & 1,
                    value & 1,
                ))

            inputs += [(0, 0, 0)] * (n - len(ids))
            active = [
                i < len(ids)
                for i in range(n)
            ]

            decoded = self.bench.capture(
                inputs,
                active
            )

            yield {
                'bit': 0,
                'width': 1,
                'stage': 'full-adder truth test',
            }

            for lane, value in enumerate(ids):
                expected = (
                    ((value >> 2) & 1)
                    + ((value >> 1) & 1)
                    + (value & 1)
                )

                if decoded[lane]['k'] != expected:
                    raise RuntimeError(
                        f'Truth-table failure for '
                        f'{inputs[lane]}.'
                    )

                tested += 1

        return (
            f'All {tested} full-adder combinations passed.'
        )

    def run_cpu_test(self):
        try:
            result = cpu_pow_test(
                difficulty=12,
                max_nonce=1_000_000
            )

            if result['found']:
                self.say(
                    f'CPU test VALID: nonce {result["nonce"]}, '
                    f'{result["zeros"]} leading zero bits.'
                )
                self.say(
                    f'Digest: {result["digest"]}'
                )
            else:
                self.say(
                    'CPU test: no nonce found in diagnostic range.'
                )

            self.say(
                f'CPU test: {result["hashes"]:,} hashes in '
                f'{result["seconds"]:.3f} s = '
                f'{result["hash_rate"]:,.0f} H/s.'
            )
            self.status.set('CPU PoW test complete.')

        except Exception as exc:
            self.show_exception(
                'CPU PoW test failed',
                exc
            )

    def tick(self):
        if self.job is None:
            return

        try:
            event = next(self.job)
            self.last_camera = self.bench.last

            if 'stage' in event:
                self.cell_labels = [
                    event['stage']
                ] * self.bench.cells

            self.status.set(
                f'{self.mode}: {self.bench.cells} cells | '
                f'exposures {self.bench.exposures:,} | '
                f'{self.bench.exposure_seconds:.3f} s at '
                f'{EXPOSURE_TIME_MS:.1f} ms/exposure | '
                f'bit {event["bit"] + 1}/{event["width"]} | '
                f'margin {self.bench.worst_margin:.2f} | '
                f'{event.get("stage", "")}'
            )
            self.draw()
            self.after(self.delay, self.tick)

        except StopIteration as stop:
            result = stop.value
            self.finish_job()

            if self.mode == 'pow':
                if result['found']:
                    message = (
                        f'VALID PoW: nonce {result["nonce"]}, '
                        f'{result["zeros"]} leading zero bits.'
                    )
                    self.say(
                        f'Digest: {result["digest"]}'
                    )
                else:
                    message = (
                        'No valid nonce found in range.'
                    )

                message += (
                    f' | {result["hashes"]:,} SHA-256 hashes | '
                    f'{result["exposure_hash_rate"]:,.0f} H/s '
                    f'at {EXPOSURE_TIME_MS:.1f} ms/exposure | '
                    f'{result["exposures"]:,} optical exposures | '
                    f'{result["exposure_seconds"]:.3f} s '
                    f'exposure time | '
                    f'{result["exposures_per_hash"]:.3f} '
                    f'exposures/hash.'
                )

                self.say(message)
                self.status.set(message)

            elif self.mode == 'add':
                counts = {}

                for value in result:
                    counts[value] = counts.get(value, 0) + 1

                most_common = max(
                    counts.items(),
                    key=lambda item: item[1]
                )

                message = (
                    f'Optical sum: {self.x_value} + '
                    f'{self.y_value} = {most_common[0]} '
                    f'({most_common[1]}/{len(result)} lanes agree).'
                )

                self.say(message)
                self.status.set(message)

            elif self.mode == 'truth':
                self.say(str(result))
                self.status.set(str(result))

            self.show_mask(self.bench.cells)

        except Exception as exc:
            self.job = None
            self.set_ui_running(False)
            self.show_exception(
                'PoW/bench stopped',
                exc
            )

    def finish_job(self):
        self.job = None
        self.set_ui_running(False)

    def set_ui_running(self, running):
        state = 'disabled' if running else 'normal'

        for entry in self.entries:
            entry.configure(state=state)

        for button in self.buttons:
            button.configure(state=state)

    def stop(self):
        if self.job is not None:
            self.job.close()
            self.finish_job()
            self.status.set('Stopped.')
            self.say(
                'Stopped by user. No completed-run result.'
            )

    def show_mask(self, cells=None):
        try:
            if cells is None:
                cells = self.cell_count()

            path, image, meta = export_mask(
                Path(__file__).resolve().parent / 'mask',
                cells
            )

            self.say(
                f'Mask exported: {cells} cells, aperture '
                f'{meta["diameter"]:.3f} mm, {path}'
            )

            if (
                self.mask_window is not None
                and self.mask_window.winfo_exists()
            ):
                self.mask_window.destroy()

            self.mask_window = tk.Toplevel(self)
            self.mask_window.title(
                '20 x 20 mm physical mask'
            )

            self.update_mask_window(
                cells,
                image,
                meta
            )

        except Exception as exc:
            self.show_exception(
                'Mask export failed',
                exc
            )

    def draw(self):
        canvas = self.canvas
        canvas.delete('all')

        width = max(450, canvas.winfo_width())
        cells = self.display_cells
        columns = min(
            cells,
            max(1, int(width / 260))
        )
        rows = math.ceil(cells / columns)
        cell_width = (width - 24) / columns
        cell_height = 150

        canvas.configure(
            scrollregion=(
                0,
                0,
                width,
                rows * cell_height + 40
            )
        )

        for i in range(cells):
            x = 12 + (i % columns) * cell_width
            y = 12 + (i // columns) * cell_height

            canvas.create_rectangle(
                x,
                y,
                x + cell_width - 10,
                y + cell_height - 10,
                fill='#182338',
                outline='#475569'
            )

            title = f'Cell {i + 1}'

            if (
                i < len(self.cell_labels)
                and self.cell_labels[i]
            ):
                title += f'  {self.cell_labels[i]}'

            canvas.create_text(
                x + 12,
                y + 18,
                text=title,
                fill='white',
                anchor='w'
            )

            data = self.last_camera

            if data and i < len(data['rows']):
                row = data['rows'][i]
                decoded = data['decoded'][i]

                for j, (label, key) in enumerate([
                    ('signal', 'signal'),
                    ('ref A', 'refa'),
                    ('ref B', 'refb'),
                ]):
                    yy = y + 45 + j * 22

                    canvas.create_text(
                        x + 12,
                        yy,
                        text=label,
                        fill='#94a3b8',
                        anchor='w'
                    )

                    left = x + 70
                    right = x + cell_width - 58

                    canvas.create_rectangle(
                        left,
                        yy - 5,
                        right,
                        yy + 5,
                        fill='#080d19',
                        outline=''
                    )

                    canvas.create_rectangle(
                        left,
                        yy - 5,
                        left + (right - left) * row[key],
                        yy + 5,
                        fill='#34d399',
                        outline=''
                    )

                    canvas.create_text(
                        right + 4,
                        yy,
                        text=f'{row[key]:.2f}',
                        fill='white',
                        anchor='w'
                    )

                canvas.create_text(
                    x + 12,
                    y + 125,
                    text=(
                        f'Level {decoded["level"]:.2f} -> '
                        f'sum {decoded["sum"]} '
                        f'carry {decoded["carry"]}'
                    ),
                    fill='#67e8f9',
                    anchor='w'
                )

            else:
                canvas.create_text(
                    x + 12,
                    y + 80,
                    text='Ready for exposure',
                    fill='#94a3b8',
                    anchor='w'
                )


def main():
    try:
        app = App()
        app.mainloop()
    except Exception:
        traceback.print_exc()
        input('\nPress Enter to close...')


if __name__ == '__main__':
    main()
