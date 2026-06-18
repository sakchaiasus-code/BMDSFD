"""
Beam Bending Web App - Flask Backend
"""
from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io, base64, warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# Beam Engine (same logic as beam_bending.py)
# ─────────────────────────────────────────────────────────────
class Beam:
    def __init__(self, length):
        self.L = length
        self.point_loads = []
        self.moments = []
        self.udl = []
        self.supports = []
        self._reactions = {}

    def add_point_load(self, position, magnitude):
        self.point_loads.append((position, magnitude))

    def add_moment(self, position, magnitude):
        self.moments.append((position, magnitude))

    def add_udl(self, start, end, intensity):
        self.udl.append((start, end, intensity))

    def add_support(self, position, support_type):
        self.supports.append((position, support_type.lower()))

    def _total_load(self):
        total = sum(mag for _, mag in self.point_loads)
        for s, e, w in self.udl:
            total += w * (e - s)
        return total

    def _total_moment_about(self, point):
        M = 0.0
        for pos, mag in self.point_loads:
            M += mag * (pos - point)
        for s, e, w in self.udl:
            F = w * (e - s)
            centroid = (s + e) / 2
            M += F * (centroid - point)
        for pos, mag in self.moments:
            M -= mag
        return M

    def solve_reactions(self):
        fixed = [(p, t) for p, t in self.supports if t == 'fixed']
        pins  = [(p, t) for p, t in self.supports if t in ('pin', 'roller')]
        if len(fixed) == 1 and len(pins) == 0:
            self._solve_cantilever(fixed[0][0])
        elif len(fixed) == 0 and len(pins) == 2:
            self._solve_simple_beam(pins[0][0], pins[1][0])
        else:
            raise ValueError("รองรับเฉพาะ Simply Supported หรือ Cantilever Beam")

    def _solve_simple_beam(self, a, b):
        if a > b: a, b = b, a
        span = b - a
        if span == 0: raise ValueError("Support ต้องไม่อยู่จุดเดียวกัน")
        Rb = self._total_moment_about(a) / span
        Ra = self._total_load() - Rb
        self._reactions = {a: Ra, b: Rb}

    def _solve_cantilever(self, fp):
        Ry = self._total_load()
        M_fix = -self._total_moment_about(fp)
        self._reactions = {fp: {'Ry': Ry, 'M': M_fix}}

    def _get_critical_points(self):
        pts = set([0.0, self.L])
        for p, _ in self.point_loads: pts.add(p)
        for p, _ in self.moments: pts.add(p)
        for s, e, _ in self.udl: pts.add(s); pts.add(e)
        for p, _ in self.supports: pts.add(p)
        return sorted(pts)

    def _shear_moment_at(self, x, is_cantilever):
        V = 0.0; M_val = 0.0
        for pos, react in self._reactions.items():
            if pos <= x:
                if isinstance(react, dict):
                    V     += react['Ry']
                    M_val += react['M'] + react['Ry'] * (x - pos)
                else:
                    V     += react
                    M_val += react * (x - pos)
        for pos, mag in self.point_loads:
            if pos <= x:
                V     -= mag
                M_val -= mag * (x - pos)
        for s, e, w in self.udl:
            if s <= x:
                eff_e = min(e, x); length = eff_e - s
                F = w * length; centroid = s + length / 2
                V     -= F
                M_val -= F * (x - centroid)
        for pos, mag in self.moments:
            if pos <= x:
                M_val += mag
        return V, M_val

    def compute_diagrams(self, n=1000):
        self.solve_reactions()
        crits = self._get_critical_points()
        xs = [0.0]
        for i in range(len(crits) - 1):
            xs += list(np.linspace(crits[i], crits[i+1], max(int(n/len(crits)), 30)))
        xs.append(self.L)
        xs = np.unique(np.array(sorted(xs)))
        is_cant = any(t == 'fixed' for _, t in self.supports)
        V = np.zeros_like(xs); M = np.zeros_like(xs)
        for i, x in enumerate(xs):
            V[i], M[i] = self._shear_moment_at(x, is_cant)
        self.xs = xs; self.V = V; self.M = M
        return xs, V, M

    def get_reactions_text(self):
        lines = []
        for pos, react in self._reactions.items():
            if isinstance(react, dict):
                lines.append(f"x = {pos} m  →  Ry = {react['Ry']:.3f} kN,  M_fix = {react['M']:.3f} kN·m")
            else:
                lines.append(f"x = {pos} m  →  R = {react:.3f} kN")
        return lines

# ─────────────────────────────────────────────────────────────
# Plot helper
# ─────────────────────────────────────────────────────────────
def make_plot(beam, title):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9),
                             gridspec_kw={'height_ratios': [1.2, 1, 1]},
                             facecolor='#1e1e2e')
    fig.suptitle(title, fontsize=13, fontweight='bold', color='white', y=0.99)

    for ax in axes:
        ax.set_facecolor('#2a2a3e')
        ax.tick_params(colors='#cdd6f4')
        ax.xaxis.label.set_color('#cdd6f4')
        ax.yaxis.label.set_color('#cdd6f4')
        ax.title.set_color('#cdd6f4')
        for spine in ax.spines.values():
            spine.set_edgecolor('#45475a')

    _plot_beam_diagram(axes[0], beam)
    _plot_sfd(axes[1], beam)
    _plot_bmd(axes[2], beam)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def _plot_beam_diagram(ax, beam):
    L = beam.L
    ax.set_title("Beam Diagram", fontsize=10, fontweight='bold')
    ax.set_xlim(-0.05*L, 1.12*L)
    ax.set_ylim(-1.6, 2.6)
    ax.axhline(0, color='#c0a060', linewidth=7, solid_capstyle='round', zorder=2)
    ax.set_yticks([])
    ax.set_xlabel("Position (m)")

    for s, e, w in beam.udl:
        n = max(int((e-s)/L*15), 3)
        for xi in np.linspace(s, e, n):
            ax.annotate('', xy=(xi, 0.05), xytext=(xi, 0.85),
                        arrowprops=dict(arrowstyle='->', color='#89b4fa', lw=1.5))
        ax.fill_between([s, e], [0.85]*2, [0.05]*2, alpha=0.25, color='#89b4fa')
        ax.text((s+e)/2, 0.95, f"w={w} kN/m", ha='center', va='bottom',
                fontsize=8, color='#89b4fa')

    for pos, mag in beam.point_loads:
        d = -1 if mag > 0 else 1
        ax.annotate('', xy=(pos, 0), xytext=(pos, d*1.15),
                    arrowprops=dict(arrowstyle='->', color='#f38ba8', lw=2.5))
        ax.text(pos, d*1.32, f"{abs(mag)} kN", ha='center',
                va='bottom' if d < 0 else 'top', fontsize=8, color='#f38ba8')

    for pos, mag in beam.moments:
        arc = mpatches.Arc((pos, 0.3), 0.5, 0.5, angle=0,
                           theta1=0, theta2=270, color='#cba6f7', lw=2)
        ax.add_patch(arc)
        ax.text(pos+0.15, 0.65, f"M={mag} kN·m", ha='left', fontsize=7.5, color='#cba6f7')

    for pos, stype in beam.supports:
        if stype in ('pin', 'roller'):
            tri = plt.Polygon([[pos,0],[pos-0.18,-0.55],[pos+0.18,-0.55]],
                              closed=True, color='#a6e3a1', zorder=3)
            ax.add_patch(tri)
            if stype == 'roller':
                ax.plot([pos-0.22, pos+0.22], [-0.6,-0.6], color='#a6e3a1', lw=2)
        elif stype == 'fixed':
            ax.fill_betweenx([-0.55, 0.55], [pos-0.15]*2, [pos]*2,
                             color='#a6e3a1', alpha=0.7)
            for yi in np.linspace(-0.55, 0.55, 6):
                ax.plot([pos-0.28, pos-0.15], [yi, yi+0.1], color='#a6e3a1', lw=1)

    for pos, react in beam._reactions.items():
        ry = react['Ry'] if isinstance(react, dict) else react
        lbl = f"R={ry:.2f} kN"
        ax.annotate('', xy=(pos,0), xytext=(pos,-1.1),
                    arrowprops=dict(arrowstyle='->', color='#a6e3a1', lw=2.5))
        ax.text(pos, -1.35, lbl, ha='center', fontsize=7.5, color='#a6e3a1')

def _annotate_extremes(ax, xs, vals, label, L):
    for idx, c in [(np.argmax(vals),'#89dceb'),(np.argmin(vals),'#fab387')]:
        v = vals[idx]
        if abs(v) > 1e-6:
            ax.annotate(f"{label}={v:.2f}", xy=(xs[idx], v),
                        xytext=(xs[idx]+L*0.03, v*1.1+0.01),
                        fontsize=7.5, color=c,
                        arrowprops=dict(arrowstyle='->', color=c, lw=1))

def _plot_sfd(ax, beam):
    ax.set_title("Shear Force Diagram (SFD)", fontsize=10, fontweight='bold')
    ax.fill_between(beam.xs, beam.V, 0, where=(beam.V>=0),
                    color='#89b4fa', alpha=0.5, label='V > 0')
    ax.fill_between(beam.xs, beam.V, 0, where=(beam.V<0),
                    color='#f38ba8', alpha=0.5, label='V < 0')
    ax.plot(beam.xs, beam.V, color='white', linewidth=1.5)
    ax.axhline(0, color='#6c7086', lw=0.8, ls='--')
    ax.set_xlim(0, beam.L); ax.set_xlabel("Position (m)")
    ax.set_ylabel("Shear Force (kN)")
    ax.legend(fontsize=8, loc='upper right', facecolor='#313244', labelcolor='#cdd6f4')
    ax.grid(True, alpha=0.2, color='#6c7086')
    _annotate_extremes(ax, beam.xs, beam.V, "V", beam.L)

def _plot_bmd(ax, beam):
    ax.set_title("Bending Moment Diagram (BMD)", fontsize=10, fontweight='bold')
    ax.fill_between(beam.xs, beam.M, 0, where=(beam.M>=0),
                    color='#a6e3a1', alpha=0.5, label='M > 0 (sagging)')
    ax.fill_between(beam.xs, beam.M, 0, where=(beam.M<0),
                    color='#fab387', alpha=0.5, label='M < 0 (hogging)')
    ax.plot(beam.xs, beam.M, color='white', linewidth=1.5)
    ax.axhline(0, color='#6c7086', lw=0.8, ls='--')
    ax.set_xlim(0, beam.L); ax.set_xlabel("Position (m)")
    ax.set_ylabel("Bending Moment (kN·m)")
    ax.legend(fontsize=8, loc='upper right', facecolor='#313244', labelcolor='#cdd6f4')
    ax.grid(True, alpha=0.2, color='#6c7086')
    _annotate_extremes(ax, beam.xs, beam.M, "M", beam.L)

# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

def validate_input(data, length, btype):
    """ตรวจสอบ input ก่อนคำนวณ — คืน list ของ error messages (Thai)"""
    errors = []

    if length <= 0:
        errors.append("ความยาวคาน L ต้องมากกว่า 0 m")
    if length > 10000:
        errors.append("ความยาวคานไม่ควรเกิน 10,000 m")

    if btype == 'simply':
        try:
            a = float(data['pin_pos'])
            b = float(data['roller_pos'])
            if a == b:
                errors.append("ตำแหน่ง Pin และ Roller ต้องไม่เหมือนกัน")
            if not (0 <= a <= length):
                errors.append(f"ตำแหน่ง Pin ({a} m) ต้องอยู่ในช่วง 0 ถึง {length} m")
            if not (0 <= b <= length):
                errors.append(f"ตำแหน่ง Roller ({b} m) ต้องอยู่ในช่วง 0 ถึง {length} m")
        except (ValueError, KeyError):
            errors.append("ตำแหน่ง Pin/Roller ไม่ถูกต้อง — กรุณากรอกตัวเลข")

    for i, pl in enumerate(data.get('point_loads', []), 1):
        try:
            pos = float(pl['pos']); mag = float(pl['mag'])
            if not (0 <= pos <= length):
                errors.append(f"Point Load #{i}: ตำแหน่ง {pos} m อยู่นอกคาน (0 ถึง {length} m)")
            if mag == 0:
                errors.append(f"Point Load #{i}: ขนาดแรงต้องไม่เป็น 0")
        except (ValueError, KeyError):
            errors.append(f"Point Load #{i}: ค่าไม่ถูกต้อง — กรุณากรอกตัวเลข")

    for i, ul in enumerate(data.get('udl', []), 1):
        try:
            s = float(ul['start']); e = float(ul['end']); w = float(ul['w'])
            if s >= e:
                errors.append(f"UDL #{i}: ตำแหน่งเริ่มต้น ({s}) ต้องน้อยกว่าสิ้นสุด ({e})")
            if not (0 <= s <= length and 0 <= e <= length):
                errors.append(f"UDL #{i}: ช่วง {s}–{e} m อยู่นอกคาน (0 ถึง {length} m)")
            if w == 0:
                errors.append(f"UDL #{i}: ความเข้ม w ต้องไม่เป็น 0")
        except (ValueError, KeyError):
            errors.append(f"UDL #{i}: ค่าไม่ถูกต้อง — กรุณากรอกตัวเลข")

    for i, mo in enumerate(data.get('moments', []), 1):
        try:
            pos = float(mo['pos']); mag = float(mo['mag'])
            if not (0 <= pos <= length):
                errors.append(f"Moment #{i}: ตำแหน่ง {pos} m อยู่นอกคาน (0 ถึง {length} m)")
            if mag == 0:
                errors.append(f"Moment #{i}: ขนาด Moment ต้องไม่เป็น 0")
        except (ValueError, KeyError):
            errors.append(f"Moment #{i}: ค่าไม่ถูกต้อง — กรุณากรอกตัวเลข")

    total_loads = (len(data.get('point_loads', [])) +
                   len(data.get('udl', [])) +
                   len(data.get('moments', [])))
    if total_loads == 0:
        errors.append("ยังไม่มีแรงกระทำ — กรุณาเพิ่ม Point Load, UDL หรือ Moment อย่างน้อย 1 รายการ")

    return errors


@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'ok': False, 'error': 'ไม่ได้รับข้อมูล — กรุณาตรวจสอบว่าเปิดผ่าน http://localhost:5000 ไม่ใช่เปิดไฟล์ HTML ตรงๆ'})

        length = float(data.get('length', 0))
        btype  = data.get('beam_type', 'simply')
        title  = data.get('title', 'Beam Analysis') or 'Beam Analysis'

        # ── validate ──
        errs = validate_input(data, length, btype)
        if errs:
            return jsonify({'ok': False, 'error': '\n'.join(f'• {e}' for e in errs)})

        beam = Beam(length)

        if btype == 'simply':
            beam.add_support(float(data['pin_pos']),    'pin')
            beam.add_support(float(data['roller_pos']), 'roller')
        else:
            fp = 0.0 if data.get('fixed_side', 'left') == 'left' else length
            beam.add_support(fp, 'fixed')

        for pl in data.get('point_loads', []):
            beam.add_point_load(float(pl['pos']), float(pl['mag']))
        for ul in data.get('udl', []):
            beam.add_udl(float(ul['start']), float(ul['end']), float(ul['w']))
        for mo in data.get('moments', []):
            beam.add_moment(float(mo['pos']), float(mo['mag']))

        beam.compute_diagrams()
        img = make_plot(beam, title)
        reactions = beam.get_reactions_text()

        v_max  = float(np.max(np.abs(beam.V)))
        m_max  = float(np.max(np.abs(beam.M)))
        x_mmax = float(beam.xs[np.argmax(np.abs(beam.M))])

        return jsonify({
            'ok': True, 'image': img, 'reactions': reactions,
            'v_max': round(v_max, 3), 'm_max': round(m_max, 3), 'x_mmax': round(x_mmax, 3),
        })

    except ValueError as e:
        return jsonify({'ok': False, 'error': f'ค่าที่กรอกไม่ถูกต้อง: {e}'})
    except Exception as e:
        return jsonify({'ok': False, 'error': f'เกิดข้อผิดพลาด: {e}'})

if __name__ == '__main__':
    print("=" * 50)
    print("  Beam Bending Web App")
    print("  เปิดเบราว์เซอร์ที่: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
