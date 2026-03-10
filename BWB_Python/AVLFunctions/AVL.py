import numpy as np
import subprocess
import re
from pathlib import Path
import tempfile
import os
import time
from typing import List, Tuple, Optional
import random
import uuid

def generate_avl_file(
    WS_spans,
    WS_rootcs,
    WS_tipcs,
    WS_sweeps,
    WS_dihedrals,
    output_filename="bwb_generated.avl",
    case_title="BWB_Python_Generated",
    mach=0.85,
    Sref=800.0,
    Cref=28.7,
    Bref=None,
    Xcg=20.0,
    airfoil_inboard="naca25111.dat",
    airfoil_outboard="SC0410.dat",
    fin_root_chord=3.72,
    fin_tip_chord=1.39,
    fin_y_root=4.10,
    fin_z_tip=3.91,
    fin_x_tip_offset=2.33,
    panel_chordwise=16,
    panel_spanwise=48,
    elevon_start_section=1          # 0-based index — section 1 = after first segment
):
    """
    Generates a basic BWB AVL file. 
    Note: airfoil files (naca25111.dat, mh78.dat) must exist in the same folder as the generated .avl
    or in AVL's search path.
    """
    n_sections = len(WS_spans)

    # Cumulative span stations (Yle)
    y_stations = np.zeros(n_sections + 1)
    y_stations[1:] = np.cumsum(WS_spans)

    # Cumulative leading-edge x positions
    x_le = np.zeros(n_sections + 1)
    for i in range(n_sections):
        sweep_rad = np.deg2rad(WS_sweeps[i])
        dx = WS_spans[i] * np.tan(sweep_rad)
        x_le[i + 1] = x_le[i] + dx

    # Cumulative z positions (linear dihedral per segment)
    z_le = np.zeros(n_sections + 1)
    for i in range(n_sections):
        dih_rad = np.deg2rad(WS_dihedrals[i])
        dz = WS_spans[i] * np.tan(dih_rad)
        z_le[i + 1] = z_le[i] + dz

    if Bref is None:
        Bref = 2.0 * y_stations[-1]

    lines = []

    lines.append(case_title)
    lines.append(f"{mach:.3f}")
    lines.append("0  0  0")           # symmetry flags
    lines.append("")
    lines.append(f"{Sref:8.1f}   {Cref:6.2f}   {Bref:6.2f}")
    lines.append(f"{Xcg:6.2f}   0.0     0.0")   # Xcg (scalar now)
    lines.append("")

    lines.append("!=========================================================")
    lines.append("! MAIN WING / BWB")
    lines.append("!=========================================================")
    lines.append("")
    lines.append("SURFACE")
    lines.append("MainWing")
    lines.append(f"{panel_chordwise}  1.0   {panel_spanwise}  1.0")
    lines.append("")
    lines.append("YDUPLICATE")
    lines.append("0.0")
    lines.append("")

    for i in range(n_sections + 1):
        x = x_le[i]
        y = y_stations[i]
        z = z_le[i]

        if i < n_sections:
            chord = WS_rootcs[i] if i == 0 else WS_tipcs[i-1]
        else:
            chord = WS_tipcs[-1]

        airfoil = airfoil_inboard if i <= elevon_start_section else airfoil_outboard

        comment = ""
        if i == 0:           comment = "! Root section"
        elif i == n_sections: comment = "! Tip section"
        else:                comment = f"! Section {i}"

        lines.append("SECTION")
        lines.append(f"  {x:9.4f}  {y:9.4f}  {z:9.4f}  {chord:8.2f}   0.0    {comment}")
        lines.append("AFILE")
        lines.append(airfoil)

        # Add elevon control only on outboard sections (after start)
        if i > elevon_start_section and i < n_sections:
            lines.append("CONTROL")
            lines.append("elevon  1.0  0.70  0.0   1  0  0")   # name, gain, hinge, sign, axis

        lines.append("")

    # Vertical fins (twin canted — simplified)
    lines.append("!=========================================================")
    lines.append("! TWIN CANTED VERTICAL FINS")
    lines.append("!=========================================================")
    lines.append("")
    lines.append("SURFACE")
    lines.append("VFinLeft")   # only left → YDUPLICATE mirrors
    lines.append("8  1.0   12  1.0")
    lines.append("")
    lines.append("YDUPLICATE")
    lines.append("0.0")
    lines.append("")
    lines.append("SECTION")
    lines.append(f"  35.5900   {fin_y_root:6.4f}   0.0000   {fin_root_chord:6.4f}   0.0")
    lines.append("NACA")
    lines.append("0012")
    lines.append("")
    x_fin_tip = 35.5900 + fin_x_tip_offset
    y_fin_tip = fin_y_root + 2.2565   # approximate span of fin
    lines.append("SECTION")
    lines.append(f"  {x_fin_tip:7.4f}   {y_fin_tip:6.4f}   {fin_z_tip:6.4f}   {fin_tip_chord:6.4f}   0.0")
    lines.append("NACA")
    lines.append("0012")
    lines.append("")

    # Write file
    output_path = Path(output_filename)
    with output_path.open('w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

    return output_path.absolute()

def wait_for_patterns(
    proc: subprocess.Popen,
    success_patterns: List[str],
    timeout: float = 15.0,
    poll_interval: float = 0.15,
    max_lines: int = 200
) -> Tuple[bool, str]:
    start_time = time.time()
    buffer = ""
    line_count = 0

    while time.time() - start_time < timeout and line_count < max_lines:
        line = proc.stdout.readline()
        if line:
            print("AVL:", line.rstrip())          # ← Print EVERY line immediately
            buffer += line
            line_count += 1
            for pattern in success_patterns:
                if pattern.lower() in buffer.lower():
                    return True, buffer
        else:
            time.sleep(poll_interval)

    return False, buffer


def _run_avl_isolated(
    avl_file_path: Path,
    avl_exe_path: Path,
    command_sequence: list,
    output_file: Path,
    timeout_seconds: int,
    max_retries: int = 3,
) -> str:

    avl_cwd = avl_file_path.parent

    run_file_path = avl_cwd / f"avl_commands_{uuid.uuid4().hex[:8]}.run"
    with open(run_file_path, 'w', encoding='utf-8') as f:
        for cmd in command_sequence:
            f.write(cmd + '\n')

    def run_avl():
        subprocess.run(
            [str(avl_exe_path)],
            stdin=open(run_file_path, 'r'),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(avl_cwd),
            timeout=timeout_seconds,
            bufsize=1,
            universal_newlines=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        return

    for attempt in range(max_retries):
        run_avl()

        if output_file.exists():
            os.remove(run_file_path)
            with open(output_file, 'r', encoding='utf-8') as f:
                return f.read()

        # Exponential backoff before retry
        wait = (attempt + 1) * random.uniform(0.3, 0.7)
        print(f"  AVL output missing, retry {attempt + 1}/{max_retries} (wait {wait:.2f}s)")
        time.sleep(wait)
    os.remove(run_file_path)
    
    raise FileNotFoundError(
        f"AVL failed to write output after {max_retries} attempts: {output_file.name}"
    )


def get_neutral_point_from_avl(
    base_name: str | Path,
    alpha_cruise: float = 2.0,
    CL_cruise: float = 0.185,
    rho: float = 0.3,
    avl_executable: str = r"C:\Users\ajbur\OneDrive\Desktop\School\MAE FILES\BURNS\MAE 155A\BWB_Python\AVLFunctions\avl352.exe",
    timeout_seconds: int = 60,
) -> Optional[float]:

    avl_file_path = Path(f'{base_name}.avl').resolve()
    if not avl_file_path.is_file():
        raise FileNotFoundError(f"AVL input file not found: {avl_file_path}")

    avl_exe_path = Path(avl_executable).resolve()
    if not avl_exe_path.is_file():
        raise FileNotFoundError(f"AVL executable not found: {avl_exe_path}")

    np_file = avl_file_path.parent / f'{Path(base_name).name}_np.txt'

    command_sequence = [
        f'LOAD {avl_file_path.name}',
        '',
        'OPER',
        #f'a a {alpha_cruise}',
        f'a C {CL_cruise}',
        'M',
        'V 295.7',
        f'D {rho}',
        '',
        'x',
        'ST',
        np_file.name,           # bare filename — AVL writes relative to cwd
    ]

    text = _run_avl_isolated(
        avl_file_path, avl_exe_path,
        command_sequence, np_file,
        timeout_seconds
    )

    match = re.search(r'Neutral\s+point\s+Xnp\s*=\s*([+-]?\d*\.?\d+)', text, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not find Neutral point in AVL output:\n{text[:500]}")

    return float(match.group(1))


def get_root_moment_from_avl(
    base_name: str | Path,
    alpha_cruise: float = 2.0,
    CL_cruise: float = 0.185,
    WS_spans: list = [4.08, 6.5, 19, 2],
    rho: float = 0.3,
    avl_executable: str = r"C:\Users\ajbur\OneDrive\Desktop\School\MAE FILES\BURNS\MAE 155A\BWB_Python\AVLFunctions\avl352.exe",
    timeout_seconds: int = 60,
) -> np.ndarray:

    avl_file_path = Path(f'{base_name}.avl').resolve()
    if not avl_file_path.is_file():
        raise FileNotFoundError(f"AVL input file not found: {avl_file_path}")

    avl_exe_path = Path(avl_executable).resolve()
    if not avl_exe_path.is_file():
        raise FileNotFoundError(f"AVL executable not found: {avl_exe_path}")

    moment_file = avl_file_path.parent / f'{Path(base_name).name}_moment.txt'

    command_sequence = [
        f'LOAD {avl_file_path.name}',
        '',
        'OPER',
        #f'a a {alpha_cruise}',
        f'a C {CL_cruise}',
        'M',
        'V 295.7',
        f'D {rho}',
        '',
        'x',
        'VM',
        moment_file.name,       # bare filename — AVL writes relative to cwd
    ]

    text = _run_avl_isolated(
        avl_file_path, avl_exe_path,
        command_sequence, moment_file,
        timeout_seconds
    )

    if 'Surface' not in text:
        raise ValueError(f"AVL moment file missing Surface blocks:\n{text[:500]}")

    def parse_distribution(surface_num: int) -> tuple[np.ndarray, np.ndarray]:
        pattern = (
            rf'Surface:\s+{surface_num}\b.*?'
            r'Mx/\(q\*Bref\*Sref\)\s*\n'
            r'((?:\s+[-+]?[\d.]+\s+[-+]?[\d.E+-]+\s+[-+]?[\d.E+-]+\n)+)'
        )
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if not m:
            raise ValueError(f"Could not find Surface {surface_num} in AVL output")

        rows = []
        for line in m.group(1).strip().split('\n'):
            vals = line.split()
            if len(vals) == 3:
                rows.append((abs(float(vals[0])), float(vals[2])))

        rows.sort(key=lambda r: r[0])
        return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])

    wing_y_arr, wing_mx_arr = parse_distribution(surface_num=1)

    # Cumulative span stations, skip root (index 0)
    y_stations = np.zeros(len(WS_spans) + 1)
    y_stations[1:] = np.cumsum(WS_spans)
    bref = sum(WS_spans) * 2
    wing_eta = [2 * y / bref for y in y_stations[:-1]]

    return np.interp(wing_eta, wing_y_arr, wing_mx_arr)