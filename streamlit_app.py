"""
Beam Bending Web App — Streamlit Version
BMD & SFD Calculator
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# Beam Engine
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
# Plot
# ─────────────────────────────────────────────────────────────
def make_figure(beam, title):
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
    return fig

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
# Streamlit UI
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BMD & SFD Calculator",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ Beam Bending Calculator")
st.caption("คำนวณ Shear Force Diagram (SFD) และ Bending Moment Diagram (BMD)")

# ── Sidebar: Input ──────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ ข้อมูลคาน")

    title = st.text_input("ชื่อโปรเจกต์ / Title", value="Beam Analysis")
    beam_type = st.radio("ประเภทคาน", ["Simply Supported", "Cantilever"])
    length = st.number_input("ความยาวคาน L (m)", min_value=0.1, max_value=10000.0,
                             value=10.0, step=0.5)

    st.divider()
    st.subheader("📌 Support")
    if beam_type == "Simply Supported":
        pin_pos    = st.number_input("ตำแหน่ง Pin (m)",    min_value=0.0, max_value=float(length), value=0.0)
        roller_pos = st.number_input("ตำแหน่ง Roller (m)", min_value=0.0, max_value=float(length), value=float(length))
    else:
        fixed_side = st.radio("Fixed end อยู่ที่", ["ซ้าย (x=0)", f"ขวา (x={length})"])

    st.divider()
    st.subheader("⬇️ Point Loads")
    n_pl = st.number_input("จำนวน Point Load", min_value=0, max_value=10, value=1, step=1)
    point_loads = []
    for i in range(int(n_pl)):
        c1, c2 = st.columns(2)
        pos = c1.number_input(f"PL#{i+1} x (m)", key=f"pl_pos_{i}",
                              min_value=0.0, max_value=float(length), value=float(length)/2)
        mag = c2.number_input(f"PL#{i+1} F (kN)", key=f"pl_mag_{i}",
                              value=10.0, help="+ลง / -ขึ้น")
        point_loads.append((pos, mag))

    st.divider()
    st.subheader("📏 Uniform Distributed Load (UDL)")
    n_udl = st.number_input("จำนวน UDL", min_value=0, max_value=5, value=0, step=1)
    udl_loads = []
    for i in range(int(n_udl)):
        c1, c2, c3 = st.columns(3)
        s = c1.number_input(f"UDL#{i+1} เริ่ม (m)", key=f"udl_s_{i}",
                            min_value=0.0, max_value=float(length), value=0.0)
        e = c2.number_input(f"UDL#{i+1} สิ้นสุด (m)", key=f"udl_e_{i}",
                            min_value=0.0, max_value=float(length), value=float(length))
        w = c3.number_input(f"UDL#{i+1} w (kN/m)", key=f"udl_w_{i}",
                            value=5.0, help="+ลง / -ขึ้น")
        udl_loads.append((s, e, w))

    st.divider()
    st.subheader("🔄 Applied Moment")
    n_mom = st.number_input("จำนวน Moment", min_value=0, max_value=5, value=0, step=1)
    moments = []
    for i in range(int(n_mom)):
        c1, c2 = st.columns(2)
        pos = c1.number_input(f"M#{i+1} x (m)", key=f"mom_pos_{i}",
                              min_value=0.0, max_value=float(length), value=0.0)
        mag = c2.number_input(f"M#{i+1} M (kN·m)", key=f"mom_mag_{i}",
                              value=10.0, help="+ทวนเข็ม / -ตามเข็ม")
        moments.append((pos, mag))

    st.divider()
    calc_btn = st.button("🧮 คำนวณ", type="primary", use_container_width=True)

# ── Main Area: Result ────────────────────────────────────────
if calc_btn:
    try:
        # Validate
        errors = []
        if length <= 0:
            errors.append("ความยาวคาน L ต้องมากกว่า 0 m")
        if beam_type == "Simply Supported" and pin_pos == roller_pos:
            errors.append("ตำแหน่ง Pin และ Roller ต้องไม่เหมือนกัน")
        total_loads = len(point_loads) + len(udl_loads) + len(moments)
        if total_loads == 0:
            errors.append("กรุณาเพิ่มแรงกระทำอย่างน้อย 1 รายการ (Point Load, UDL หรือ Moment)")
        for i, (s, e, w) in enumerate(udl_loads, 1):
            if s >= e:
                errors.append(f"UDL #{i}: ตำแหน่งเริ่มต้น ({s}) ต้องน้อยกว่าสิ้นสุด ({e})")

        if errors:
            for err in errors:
                st.error(f"❌ {err}")
        else:
            # Build beam
            beam = Beam(length)
            if beam_type == "Simply Supported":
                beam.add_support(pin_pos, 'pin')
                beam.add_support(roller_pos, 'roller')
            else:
                fp = 0.0 if "ซ้าย" in fixed_side else length
                beam.add_support(fp, 'fixed')

            for pos, mag in point_loads:
                beam.add_point_load(pos, mag)
            for s, e, w in udl_loads:
                beam.add_udl(s, e, w)
            for pos, mag in moments:
                beam.add_moment(pos, mag)

            beam.compute_diagrams()

            # Results summary
            v_max  = float(np.max(np.abs(beam.V)))
            m_max  = float(np.max(np.abs(beam.M)))
            x_mmax = float(beam.xs[np.argmax(np.abs(beam.M))])

            col1, col2, col3 = st.columns(3)
            col1.metric("V_max (Shear Force)", f"{v_max:.3f} kN")
            col2.metric("M_max (Bending Moment)", f"{m_max:.3f} kN·m")
            col3.metric("ตำแหน่ง M_max", f"x = {x_mmax:.3f} m")

            st.subheader("📊 Reactions")
            for r in beam.get_reactions_text():
                st.info(r)

            # Plot
            fig = make_figure(beam, title or "Beam Analysis")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    except ValueError as e:
        st.error(f"❌ ค่าที่กรอกไม่ถูกต้อง: {e}")
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {e}")

else:
    st.info("👈 กรอกข้อมูลคานในแถบซ้าย แล้วกด **คำนวณ**")
    st.markdown("""
    **วิธีใช้งาน:**
    1. เลือกประเภทคาน (Simply Supported / Cantilever)
    2. กำหนดความยาวและตำแหน่ง Support
    3. เพิ่ม Point Load, UDL หรือ Moment
    4. กดปุ่ม **คำนวณ** เพื่อดูผล
    """)
