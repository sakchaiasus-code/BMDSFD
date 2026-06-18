"""
Beam Bending Diagram Calculator
================================
คำนวณ Shear Force Diagram (SFD) และ Bending Moment Diagram (BMD)
รองรับ: Point Load, Distributed Load (UDL), Applied Moment, Supports (Pin, Roller, Fixed)
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # เปลี่ยนเป็น 'Agg' ถ้าไม่มี display หรือรันใน server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')


class Beam:
    def __init__(self, length: float):
        """
        Parameters
        ----------
        length : float  ความยาวคาน (m)
        """
        self.L = length
        self.point_loads = []       # (position, magnitude)  + ลง, - ขึ้น
        self.moments = []           # (position, magnitude)  + ทวนเข็ม, - ตามเข็ม
        self.udl = []               # (start, end, intensity)  + ลง
        self.supports = []          # (position, type)  'pin','roller','fixed'
        self._reactions = {}

    # ------------------------------------------------------------------ add
    def add_point_load(self, position: float, magnitude: float):
        """เพิ่ม Point Load  magnitude > 0 = ลง"""
        self.point_loads.append((position, magnitude))

    def add_moment(self, position: float, magnitude: float):
        """เพิ่ม Applied Moment  magnitude > 0 = ทวนเข็มนาฬิกา"""
        self.moments.append((position, magnitude))

    def add_udl(self, start: float, end: float, intensity: float):
        """เพิ่ม Uniform Distributed Load  intensity > 0 = ลง"""
        self.udl.append((start, end, intensity))

    def add_support(self, position: float, support_type: str):
        """เพิ่ม Support  type: 'pin', 'roller', 'fixed'"""
        self.supports.append((position, support_type.lower()))

    # ---------------------------------------------------------- solve reactions
    def solve_reactions(self):
        """แก้ปัญหา reactions (simple beam / cantilever)"""
        support_types = [(p, t) for p, t in self.supports]

        # Fixed end (cantilever)
        fixed = [(p, t) for p, t in support_types if t == 'fixed']
        pins   = [(p, t) for p, t in support_types if t in ('pin', 'roller')]

        if len(fixed) == 1 and len(pins) == 0:
            self._solve_cantilever(fixed[0][0])
        elif len(fixed) == 0 and len(pins) == 2:
            self._solve_simple_beam(pins[0][0], pins[1][0])
        elif len(fixed) == 0 and len(pins) == 1:
            # treat as propped – raise if over-constrained
            raise ValueError("ต้องการ support อย่างน้อย 2 จุดสำหรับ simply supported beam")
        else:
            raise ValueError("รองรับเฉพาะ Simply Supported Beam หรือ Cantilever Beam")

    def _total_load(self):
        total = sum(mag for _, mag in self.point_loads)
        for s, e, w in self.udl:
            total += w * (e - s)
        return total

    def _total_moment_about(self, point):
        """Moment จาก external loads รอบจุด point (ตามเข็ม = +)"""
        M = 0.0
        for pos, mag in self.point_loads:
            M += mag * (pos - point)          # force × arm  (+ ลง × arm ขวา = ตามเข็ม)
        for s, e, w in self.udl:
            F = w * (e - s)
            centroid = (s + e) / 2
            M += F * (centroid - point)
        for pos, mag in self.moments:
            M -= mag                            # applied moment (+ = ทวนเข็ม) → ตามเข็ม = -
        return M

    def _solve_simple_beam(self, a, b):
        """Simply supported beam – pin ที่ a, roller ที่ b"""
        if a > b:
            a, b = b, a
        span = b - a
        if span == 0:
            raise ValueError("Support ต้องไม่อยู่ที่ตำแหน่งเดียวกัน")
        Rb = self._total_moment_about(a) / span
        Ra = self._total_load() - Rb
        self._reactions = {a: Ra, b: Rb}

    def _solve_cantilever(self, fixed_pos):
        """Cantilever – fixed end ที่ fixed_pos"""
        Ry = self._total_load()
        M_fix = -self._total_moment_about(fixed_pos)   # reaction moment
        self._reactions = {fixed_pos: {'Ry': Ry, 'M': M_fix}}

    # ---------------------------------------------------------- SFD / BMD
    def _get_critical_points(self):
        pts = set([0.0, self.L])
        for p, _ in self.point_loads:
            pts.add(p)
        for p, _ in self.moments:
            pts.add(p)
        for s, e, _ in self.udl:
            pts.add(s); pts.add(e)
        for p, _ in self.supports:
            pts.add(p)
        return sorted(pts)

    def compute_diagrams(self, n_points: int = 1000):
        """คำนวณ SFD และ BMD ตลอดความยาวคาน"""
        self.solve_reactions()

        crits = self._get_critical_points()
        xs = [0.0]
        for i in range(len(crits) - 1):
            xs += list(np.linspace(crits[i], crits[i + 1], max(int(n_points / len(crits)), 20)))
        xs.append(self.L)
        xs = np.unique(np.array(sorted(xs)))

        V = np.zeros_like(xs)
        M = np.zeros_like(xs)

        # ตรวจว่าเป็น cantilever หรือ simply supported
        is_cantilever = any(t == 'fixed' for _, t in self.supports)

        for i, x in enumerate(xs):
            v, m = self._shear_moment_at(x, is_cantilever)
            V[i] = v
            M[i] = m

        self.xs = xs
        self.V  = V
        self.M  = M
        return xs, V, M

    def _shear_moment_at(self, x, is_cantilever):
        """คำนวณ V และ M ที่ตำแหน่ง x โดยตัดจากซ้าย"""
        V = 0.0
        M_val = 0.0

        # Reactions ทางซ้ายของ x
        if is_cantilever:
            for pos, react in self._reactions.items():
                if pos <= x:
                    if isinstance(react, dict):
                        V     += react['Ry']
                        M_val += react['M'] + react['Ry'] * (x - pos)
                    else:
                        V     += react
                        M_val += react * (x - pos)
        else:
            for pos, react in self._reactions.items():
                if pos <= x:
                    V     += react          # reaction ขึ้น = ลบ shear ตามเข็ม
                    M_val += react * (x - pos)

        # Point loads ทางซ้ายของ x  (ลง = +V ด้านซ้าย → ลด V)
        for pos, mag in self.point_loads:
            if pos <= x:
                V     -= mag
                M_val -= mag * (x - pos)

        # UDL ทางซ้ายของ x
        for s, e, w in self.udl:
            if s <= x:
                eff_e = min(e, x)
                length = eff_e - s
                F = w * length
                centroid = s + length / 2
                V     -= F
                M_val -= F * (x - centroid)

        # Applied moments ทางซ้ายของ x
        for pos, mag in self.moments:
            if pos <= x:
                M_val += mag   # + = ทวนเข็ม = บวก M

        return V, M_val

    # ------------------------------------------------------------------ plot
    def plot(self, title: str = "Beam Analysis"):
        self.compute_diagrams()

        fig, axes = plt.subplots(3, 1, figsize=(12, 10),
                                 gridspec_kw={'height_ratios': [1.2, 1, 1]})
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        self._plot_beam_diagram(axes[0])
        self._plot_sfd(axes[1])
        self._plot_bmd(axes[2])

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig("E:/Clauade code/beam_bending_output.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("บันทึกภาพ: beam_bending_output.png")

    def _plot_beam_diagram(self, ax):
        ax.set_title("Beam Diagram", fontsize=11, fontweight='bold')
        ax.set_xlim(-0.05 * self.L, 1.1 * self.L)
        ax.set_ylim(-1.5, 2.5)
        ax.axhline(0, color='saddlebrown', linewidth=6, solid_capstyle='round', zorder=2)
        ax.set_yticks([])
        ax.set_xlabel("Position (m)")

        # UDL
        for s, e, w in self.udl:
            n_arrows = max(int((e - s) / self.L * 15), 3)
            for xi in np.linspace(s, e, n_arrows):
                ax.annotate('', xy=(xi, 0.05), xytext=(xi, 0.9),
                            arrowprops=dict(arrowstyle='->', color='royalblue', lw=1.5))
            ax.fill_between([s, e], [0.9, 0.9], [0.05, 0.05],
                            alpha=0.2, color='royalblue')
            ax.text((s + e) / 2, 1.0, f"w={w} kN/m",
                    ha='center', va='bottom', fontsize=8, color='royalblue')

        # Point loads
        for pos, mag in self.point_loads:
            direction = -1 if mag > 0 else 1
            ax.annotate('', xy=(pos, 0), xytext=(pos, direction * 1.1),
                        arrowprops=dict(arrowstyle='->', color='crimson', lw=2))
            ax.text(pos, direction * 1.25, f"{abs(mag)} kN",
                    ha='center', va='bottom' if direction < 0 else 'top',
                    fontsize=8, color='crimson')

        # Applied moments
        for pos, mag in self.moments:
            arc = mpatches.Arc((pos, 0.3), 0.4, 0.4, angle=0,
                               theta1=0, theta2=270,
                               color='purple', lw=2)
            ax.add_patch(arc)
            ax.text(pos + 0.1, 0.6, f"M={mag} kNm",
                    ha='left', fontsize=8, color='purple')

        # Supports
        for pos, stype in self.supports:
            if stype in ('pin', 'roller'):
                triangle = plt.Polygon([[pos, 0], [pos - 0.15, -0.5], [pos + 0.15, -0.5]],
                                       closed=True, color='gray', zorder=3)
                ax.add_patch(triangle)
                if stype == 'roller':
                    ax.plot([pos - 0.2, pos + 0.2], [-0.55, -0.55],
                            color='gray', lw=2)
            elif stype == 'fixed':
                ax.fill_betweenx([-0.5, 0.5], [pos - 0.15, pos - 0.15],
                                 [pos, pos], color='gray', alpha=0.7)
                for yi in np.linspace(-0.5, 0.5, 5):
                    ax.plot([pos - 0.25, pos - 0.15], [yi, yi + 0.1],
                            color='gray', lw=1)

        # Reactions
        is_cantilever = any(t == 'fixed' for _, t in self.supports)
        for pos, react in self._reactions.items():
            if isinstance(react, dict):
                ry = react['Ry']
                label = f"Ry={ry:.2f} kN"
            else:
                ry = react
                label = f"R={ry:.2f} kN"
            color = 'darkgreen'
            ax.annotate('', xy=(pos, 0), xytext=(pos, -1.0),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
            ax.text(pos, -1.2, label, ha='center', fontsize=8, color=color)

    def _plot_sfd(self, ax):
        ax.set_title("Shear Force Diagram (SFD)", fontsize=11, fontweight='bold')
        ax.fill_between(self.xs, self.V, 0,
                        where=(self.V >= 0), color='steelblue', alpha=0.4, label='V > 0')
        ax.fill_between(self.xs, self.V, 0,
                        where=(self.V < 0), color='tomato', alpha=0.4, label='V < 0')
        ax.plot(self.xs, self.V, 'k-', linewidth=1.5)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlim(0, self.L)
        ax.set_xlabel("Position (m)")
        ax.set_ylabel("Shear Force (kN)")
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        self._annotate_extremes(ax, self.V, "V")

    def _plot_bmd(self, ax):
        ax.set_title("Bending Moment Diagram (BMD)", fontsize=11, fontweight='bold')
        ax.fill_between(self.xs, self.M, 0,
                        where=(self.M >= 0), color='limegreen', alpha=0.4, label='M > 0 (sagging)')
        ax.fill_between(self.xs, self.M, 0,
                        where=(self.M < 0), color='orange', alpha=0.4, label='M < 0 (hogging)')
        ax.plot(self.xs, self.M, 'k-', linewidth=1.5)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlim(0, self.L)
        ax.set_xlabel("Position (m)")
        ax.set_ylabel("Bending Moment (kNm)")
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        self._annotate_extremes(ax, self.M, "M")

    def _annotate_extremes(self, ax, values, label):
        idx_max = np.argmax(values)
        idx_min = np.argmin(values)
        for idx, color in [(idx_max, 'navy'), (idx_min, 'darkred')]:
            x_val = self.xs[idx]
            y_val = values[idx]
            if abs(y_val) > 1e-6:
                ax.annotate(f"{label}={y_val:.2f}",
                            xy=(x_val, y_val),
                            xytext=(x_val + self.L * 0.03, y_val * 1.1 + 0.01),
                            fontsize=7.5, color=color,
                            arrowprops=dict(arrowstyle='->', color=color, lw=1))


# ===========================================================================
# ตัวอย่างการใช้งาน
# ===========================================================================
def example_simply_supported():
    """Simply Supported Beam ที่มี Point Load และ UDL"""
    print("=" * 55)
    print("Example 1: Simply Supported Beam")
    print("  ความยาว: 10 m")
    print("  Pin ที่ x=0, Roller ที่ x=10")
    print("  Point Load 20 kN ที่ x=4")
    print("  UDL 5 kN/m ระหว่าง x=6 ถึง x=10")
    print("=" * 55)

    beam = Beam(length=10)
    beam.add_support(0, 'pin')
    beam.add_support(10, 'roller')
    beam.add_point_load(4, 20)    # 20 kN ลง ที่ x=4 m
    beam.add_udl(6, 10, 5)        # 5 kN/m ระหว่าง 6-10 m
    beam.solve_reactions()

    print("\nReactions:")
    for pos, R in beam._reactions.items():
        print(f"  x = {pos} m  =>  R = {R:.3f} kN")

    beam.plot("Example 1: Simply Supported Beam with Point Load + UDL")


def example_cantilever():
    """Cantilever Beam ที่มี Point Load ปลาย"""
    print("\n" + "=" * 55)
    print("Example 2: Cantilever Beam")
    print("  ความยาว: 6 m")
    print("  Fixed ที่ x=0")
    print("  Point Load 15 kN ที่ x=6 (ปลายอิสระ)")
    print("  UDL 3 kN/m ตลอดคาน")
    print("=" * 55)

    beam = Beam(length=6)
    beam.add_support(0, 'fixed')
    beam.add_point_load(6, 15)    # 15 kN ที่ปลาย
    beam.add_udl(0, 6, 3)         # 3 kN/m ตลอด

    beam.solve_reactions()

    r = beam._reactions[0]
    print(f"\nReactions at Fixed End (x=0):")
    print(f"  Ry   = {r['Ry']:.3f} kN  (up)")
    print(f"  M_fix= {r['M']:.3f} kNm")

    beam.plot("Example 2: Cantilever Beam with Point Load + UDL")


def example_two_point_loads():
    """Simply Supported Beam ที่มี Point Loads 2 จุด"""
    print("\n" + "=" * 55)
    print("Example 3: Simply Supported – Two Point Loads")
    print("  ความยาว: 8 m")
    print("  Pin ที่ x=0, Roller ที่ x=8")
    print("  10 kN ที่ x=2,  25 kN ที่ x=5")
    print("=" * 55)

    beam = Beam(length=8)
    beam.add_support(0, 'pin')
    beam.add_support(8, 'roller')
    beam.add_point_load(2, 10)
    beam.add_point_load(5, 25)

    beam.solve_reactions()
    print("\nReactions:")
    for pos, R in beam._reactions.items():
        print(f"  x = {pos} m  =>  R = {R:.3f} kN")

    beam.plot("Example 3: Simply Supported Beam with Two Point Loads")


# ===========================================================================
# Interactive Input
# ===========================================================================
def get_float(prompt, default=None):
    while True:
        try:
            val = input(prompt).strip()
            if val == "" and default is not None:
                return default
            return float(val)
        except ValueError:
            print("  [!] กรุณากรอกตัวเลข")

def get_int(prompt, default=None):
    while True:
        try:
            val = input(prompt).strip()
            if val == "" and default is not None:
                return default
            return int(val)
        except ValueError:
            print("  [!] กรุณากรอกจำนวนเต็ม")

def get_choice(prompt, choices):
    while True:
        val = input(prompt).strip().lower()
        if val in choices:
            return val
        print(f"  [!] กรุณาเลือก: {', '.join(choices)}")

def interactive_mode():
    print("\n" + "=" * 60)
    print("  Beam Bending Diagram Calculator - Interactive Mode")
    print("=" * 60)

    # ---- beam type ----
    print("\nเลือกประเภทคาน:")
    print("  1 = Simply Supported Beam (Pin + Roller)")
    print("  2 = Cantilever Beam (Fixed)")
    beam_type = get_choice("เลือก (1/2): ", ["1", "2"])

    length = get_float("ความยาวคาน L (m): ")
    beam = Beam(length)

    if beam_type == "1":
        print(f"\n[Support] Simply Supported")
        a = get_float(f"  ตำแหน่ง Pin (0 ถึง {length}): ", default=0)
        b = get_float(f"  ตำแหน่ง Roller (0 ถึง {length}): ", default=length)
        beam.add_support(a, 'pin')
        beam.add_support(b, 'roller')
    else:
        print(f"\n[Support] Cantilever")
        fix = get_choice("  Fixed end อยู่ที่ไหน? (left=0 / right): ", ["left", "right"])
        fp = 0 if fix == "left" else length
        beam.add_support(fp, 'fixed')

    # ---- point loads ----
    n_pl = get_int("\nจำนวน Point Load (0 ถึง 10): ", default=0)
    for i in range(n_pl):
        print(f"  Point Load #{i+1}")
        pos = get_float(f"    ตำแหน่ง x (m): ")
        mag = get_float(f"    แรง (kN, บวก=ลง, ลบ=ขึ้น): ")
        beam.add_point_load(pos, mag)

    # ---- UDL ----
    n_udl = get_int("\nจำนวน Uniform Distributed Load - UDL (0 ถึง 5): ", default=0)
    for i in range(n_udl):
        print(f"  UDL #{i+1}")
        s = get_float(f"    เริ่มต้นที่ x (m): ")
        e = get_float(f"    สิ้นสุดที่ x (m): ")
        w = get_float(f"    ความเข้ม w (kN/m, บวก=ลง): ")
        beam.add_udl(s, e, w)

    # ---- Applied Moments ----
    n_mom = get_int("\nจำนวน Applied Moment (0 ถึง 5): ", default=0)
    for i in range(n_mom):
        print(f"  Moment #{i+1}")
        pos = get_float(f"    ตำแหน่ง x (m): ")
        mag = get_float(f"    ขนาด (kNm, บวก=ทวนเข็ม): ")
        beam.add_moment(pos, mag)

    # ---- title ----
    title = input("\nชื่อกราฟ (Enter=ข้าม): ").strip()
    if not title:
        title = "Beam Analysis"

    # ---- solve & plot ----
    print("\n" + "-" * 40)
    try:
        beam.solve_reactions()
        print("Reactions:")
        for pos, react in beam._reactions.items():
            if isinstance(react, dict):
                print(f"  x = {pos} m  =>  Ry = {react['Ry']:.3f} kN,  M_fix = {react['M']:.3f} kNm")
            else:
                print(f"  x = {pos} m  =>  R = {react:.3f} kN")
        beam.plot(title)
    except Exception as ex:
        print(f"\n[ERROR] {ex}")


def show_menu():
    print("\n" + "=" * 60)
    print("  Beam Bending Diagram Calculator")
    print("=" * 60)
    print("  1 = กรอกข้อมูลเอง (Interactive)")
    print("  2 = ดูตัวอย่าง: Simply Supported + Point Load + UDL")
    print("  3 = ดูตัวอย่าง: Cantilever Beam")
    print("  4 = ดูตัวอย่าง: Simply Supported + 2 Point Loads")
    print("  0 = ออกจากโปรแกรม")
    print("-" * 60)
    return get_choice("เลือก: ", ["0", "1", "2", "3", "4"])


# ===========================================================================
if __name__ == "__main__":
    while True:
        choice = show_menu()
        if choice == "0":
            print("ออกจากโปรแกรม")
            break
        elif choice == "1":
            interactive_mode()
        elif choice == "2":
            example_simply_supported()
        elif choice == "3":
            example_cantilever()
        elif choice == "4":
            example_two_point_loads()
