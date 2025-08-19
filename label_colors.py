# label_colors.py

LABEL_COLORS = {
    'chamfer': '#e6194B',
    'through_hole': '#3cb44b',
    'triangular_passage': '#ffe119',
    'rectangular_passage': '#0082c8',
    '6sides_passage': '#f58231',
    'triangular_through_slot': '#911eb4',
    'rectangular_through_slot': '#46f0f0',
    'circular_through_slot': '#f032e6',
    'rectangular_through_step': '#d2f53c',
    '2sides_through_step': '#fabebe',
    'slanted_through_step': '#008080',
    'Oring': '#e6beff',
    'blind_hole': '#aa6e28',
    'triangular_pocket': '#fffac8',
    'rectangular_pocket': '#800000',
    '6sides_pocket': '#aaffc3',
    'circular_end_pocket': '#808000',
    'rectangular_blind_slot': '#ffd8b1',
    'v_circular_end_blind_slot': '#000080',
    'h_circular_end_blind_slot': '#808080',
    'triangular_blind_step': '#bcf60c',
    'circular_blind_step': '#9A6324',
    'rectangular_blind_step': '#469990',
    'round': '#dcbeff',
    'stock': '#4363d8',
}

# (optional) tiny helpers
def hex_to_rgb01(hx: str):
    hx = hx.lstrip('#')
    return tuple(int(hx[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def name_to_rgb01(name: str):
    return hex_to_rgb01(LABEL_COLORS[name])
