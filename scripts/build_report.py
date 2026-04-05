import json, math, warnings, os, re, base64
from collections import Counter
from itertools import combinations
from io import BytesIO

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from matplotlib.collections import LineCollection
import city2graph
import plotly.express as px
from pyvis.network import Network

warnings.filterwarnings('ignore')

# ── 0. CONFIGURAÇÃO DE CAMINHOS ─────────────────────────────────
OUT = '/mnt/HDD1TB/Documentos/testes_city2graph'
os.makedirs(os.path.join(OUT, 'assets', 'maps'), exist_ok=True)

# ── 0.1 FAVICON ─────────────────────────────────────────────────
G_fav = nx.erdos_renyi_graph(6, 0.4, seed=42)
plt.figure(figsize=(1, 1), facecolor='#0d1117')
nx.draw(G_fav, nx.spring_layout(G_fav, seed=42),
        node_size=100, node_color='#4a90b8',
        edge_color='#7d8590', width=2)
buf = BytesIO()
plt.savefig(buf, format='png', dpi=64, transparent=True)
plt.close()
b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

# ── 1. DADOS ────────────────────────────────────────────────────
print('Carregando dados...')
df         = pd.read_csv(f'{OUT}/data/survey.csv', sep=';')
axial_gdf  = gpd.read_file(f'{OUT}/data/modelo_axial_unificado.geojson')
if axial_gdf.crs is None:
    axial_gdf = axial_gdf.set_crs('EPSG:4326')
camadas_gdf = gpd.read_file(f'{OUT}/data/camadas_questionário.geojson')
if camadas_gdf.crs is None:
    camadas_gdf = camadas_gdf.set_crs('EPSG:4326')

place_cols = ['p1', 'p2', 'p3', 'p4', 'p5']
all_places = []
for col in place_cols:
    all_places.extend(df[col].dropna().str.strip().tolist())
place_freq = Counter(all_places)
camadas_gdf['freq'] = camadas_gdf['Bloco_unb'].map(lambda b: place_freq.get(b, 0))

co_pairs = Counter()
for _, row in df.iterrows():
    places = [row[c] for c in place_cols if pd.notna(row[c])]
    for a, b in combinations(sorted(set(places)), 2):
        co_pairs[(a, b)] += 1

# ── 2. GRAFO AXIAL ──────────────────────────────────────────────
print('Construindo grafo axial...')
axial_utm = axial_gdf.to_crs('EPSG:32723')
nodes_gdf, edges_gdf = city2graph.segments_to_graph(axial_utm)
G_axial = city2graph.gdf_to_nx(nodes_gdf, edges_gdf)

n_nodes   = G_axial.number_of_nodes()
n_edges   = G_axial.number_of_edges()
density   = nx.density(G_axial)
components = nx.number_connected_components(G_axial)

# ── 3. PALETA SPACE SYNTAX ──────────────────────────────────────
SS_RGB = [
    mcolors.hsv_to_rgb((240/360, 1.0, 0.90)),
    mcolors.hsv_to_rgb((195/360, 1.0, 0.95)),
    mcolors.hsv_to_rgb((120/360, 1.0, 0.85)),
    mcolors.hsv_to_rgb(( 60/360, 1.0, 0.95)),
    mcolors.hsv_to_rgb(( 25/360, 1.0, 1.00)),
    mcolors.hsv_to_rgb((  0/360, 1.0, 0.95)),
]
CMAP  = mcolors.LinearSegmentedColormap.from_list('space_syntax', SS_RGB, N=512)
BG    = '#0d1117'
TEXT  = '#e6edf3'
MUTED = '#7d8590'
DIM   = '#484f58'

# ── 4. ASPECT RATIO REAL DO CAMPUS ──────────────────────────────
utm_b      = axial_utm.total_bounds
CAMPUS_W   = utm_b[2] - utm_b[0]
CAMPUS_H   = utm_b[3] - utm_b[1]
CAMPUS_AR  = CAMPUS_W / CAMPUS_H

METRICS = {
    'INThh':   ('Integração Global (HH)',  False,
                'Facilidade de acesso. Mede quão "perto" uma via está de todas as outras.',
                'Valores > 1.0 indicam eixos centrais e integrados (vermelho). < 0.8 indicam eixos periféricos (azul).',
                'A integração revela a estrutura primária de circulação que sustenta as atividades coletivas do campus.'),
    'CONN':    ('Conectividade',            False,
                'Número de cruzamentos diretos que uma via possui.',
                'Valores altos indicam cruzamentos importantes e distribuidores de fluxo. Valores baixos indicam vias isoladas.',
                'A alta conectividade no eixo do RU e ICC Sul corrobora a importância funcional desses pontos como hubs.'),
    'control': ('Controle',                False,
                'Avalia o quanto uma via domina o acesso às suas vizinhas.',
                'Vias de alto controle são passagens obrigatórias para os eixos ao redor.',
                'Mostra áreas que controlam o fluxo de microescala, como entradas de estacionamentos.'),
    'MD':      ('Profundidade Média',       True,
                'Mede o esforço de circulação (número de mudanças de direção).',
                'Quanto MENOR o valor (vermelho na escala invertida), mais acessível o local. Valores altos (azul) indicam locais "escondidos".',
                'Áreas com alta profundidade tendem a ser mais silenciosas e segregadas.'),
    'CH':      ('Escolha (Choice)',         False,
                'Potencial de atalho. Mede o uso de uma via nos caminhos mais curtos do sistema.',
                'Valores elevados sinalizam eixos de grande fluxo de passagem (pedestres e carros).',
                'A análise de Choice permite distinguir vias de destino (Integração) de vias de passagem (Escolha).'),
}

# ── 4.1 FUNÇÕES DE MAPA PNG ─────────────────────────────────────
def normalize(series, invert=False):
    mn, mx = series.min(), series.max()
    t = (series - mn) / (mx - mn + 1e-9)
    return 1 - t if invert else t

def plot_axial_map(metric_col, title, invert=False):
    fig_w = 6.5
    fig_h = fig_w / CAMPUS_AR

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)
    ax  = fig.add_axes([0.03, 0.06, 0.94, 0.91])
    ax.set_facecolor(BG)
    ax.set_aspect('equal')

    vals      = axial_gdf[metric_col].values
    norm_vals = normalize(pd.Series(vals), invert=invert).values

    minx, miny, maxx, maxy = axial_gdf.total_bounds
    dx = (maxx - minx) * 0.028
    dy = (maxy - miny) * 0.028
    ax.set_xlim(minx - dx, maxx + dx)
    ax.set_ylim(miny - dy, maxy + dy)

    for gv in np.linspace(minx - dx, maxx + dx, 5):
        ax.axvline(gv, color='#1c2128', lw=0.22, zorder=0)
    for gv in np.linspace(miny - dy, maxy + dy, 9):
        ax.axhline(gv, color='#1c2128', lw=0.22, zorder=0)

    try:
        import contextily as cx
        cx.add_basemap(ax, crs=axial_gdf.crs,
                       source=cx.providers.CartoDB.DarkMatterNoLabels,
                       alpha=0.3, zorder=1)
    except Exception:
        pass

    for i, (_, row) in enumerate(axial_gdf.iterrows()):
        coords = list(row.geometry.coords)
        t   = float(norm_vals[i])
        col = CMAP(t)
        lw  = 1.5 + t * 2.5
        xs  = [c[0] for c in coords]
        ys  = [c[1] for c in coords]
        ax.plot(xs, ys, color=col, lw=lw,
                solid_capstyle='round', solid_joinstyle='round', zorder=2)

    freq_max = max(camadas_gdf['freq'].max(), 1)
    for _, brow in camadas_gdf.iterrows():
        lon, lat = brow.geometry.x, brow.geometry.y
        freq     = brow['freq']
        r_deg    = 0.00008 + (freq / freq_max) * 0.00065
        alpha_fill = 0.12 + (freq / freq_max) * 0.58
        ax.add_patch(plt.Circle((lon, lat), r_deg,
                                color='white', alpha=alpha_fill,
                                linewidth=0, zorder=4))
        ax.add_patch(plt.Circle((lon, lat), r_deg * 1.5,
                                fill=False, edgecolor='white',
                                alpha=0.28, lw=0.4, zorder=4))
        if freq >= 55:
            short = (brow['Bloco_unb']
                     .replace('Restaurante Universitário', 'RU')
                     .replace('Faculdade de ', 'Fac. ')
                     .replace(' (Amarelinho)', ''))[:16]
            ax.text(lon, lat - r_deg * 2.2, short,
                    ha='center', va='top', fontsize=5.5,
                    color='white', alpha=0.8, zorder=5,
                    fontfamily='monospace', fontweight='bold')

    deg_per_m = 1 / 111320
    bar_len   = 500 * deg_per_m
    sx = minx + dx * 0.3
    sy = miny + dy * 0.3
    ax.plot([sx, sx + bar_len], [sy, sy], color=MUTED, lw=0.9, zorder=6)
    ax.text(sx + bar_len / 2, sy + 0.00016, '500 m',
            ha='center', va='bottom', color=MUTED,
            fontsize=5.2, fontfamily='monospace', zorder=6)

    ax.set_title(f'{title}', color=TEXT, fontsize=9.0, fontweight='bold',
                 loc='left', pad=6, fontfamily='monospace')
    ax.set_axis_off()

    cax = fig.add_axes([0.06, 0.02, 0.78, 0.014])
    sm  = mcm.ScalarMappable(cmap=CMAP, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_ticks([0, 0.5, 1.0])
    lbl = (['alta (invertido)', 'média', 'baixa (invertido)'] if invert
           else ['baixa', 'média', 'alta'])
    cb.set_ticklabels(lbl, color=MUTED, fontsize=5.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)

    fname = os.path.join(OUT, 'assets', 'maps', f'mapa_{metric_col.lower()}.png')
    fig.savefig(fname, dpi=160, facecolor=BG, bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)
    print(f'  PNG: {fname}')
    return fname

print('Gerando mapas PNG...')
png_files = {}
for col, info in METRICS.items():
    png_files[col] = plot_axial_map(col, info[0], invert=info[1])

# ── 5. PYVIS ────────────────────────────────────────────────────
print('Gerando rede PyVis...')
top_set = set(place_freq.keys())
G_cof   = nx.Graph()
for p in top_set:
    G_cof.add_node(p, freq=place_freq[p])
for (a, b), w in co_pairs.items():
    if a in top_set and b in top_set and w >= 4:
        G_cof.add_edge(a, b, weight=w)

betweenness = nx.betweenness_centrality(G_cof, weight='weight')
freq_max_g  = max(place_freq[p] for p in top_set)

def freq_color(freq):
    t = freq / freq_max_g
    if t < 0.5:
        t2 = t * 2
        r = int(60  + t2 * (180 - 60))
        g = int(90  + t2 * (150 - 90))
        b = int(130 + t2 * (80  - 130))
    else:
        t2 = (t - 0.5) * 2
        r = int(180 + t2 * (220 - 180))
        g = int(150 - t2 * 150)
        b = int(80  - t2 * 80)
    return f'#{r:02x}{g:02x}{b:02x}'

net = Network(height='600px', width='100%', bgcolor=BG,
              font_color=TEXT, directed=False)
net.set_options(json.dumps({
    "nodes": {
        "font": {"size": 14, "face": "IBM Plex Mono, monospace",
                 "color": "#c9d1d9", "strokeWidth": 2, "strokeColor": "#000000"},
        "borderWidth": 1, "borderWidthSelected": 2,
    },
    "edges": {
        "color": {"color": "#2a3545", "highlight": "#4a90b8"},
        "smooth": {"type": "continuous"},
    },
    "physics": {
        "forceAtlas2Based": {
            "gravitationalConstant": -150,
            "centralGravity": 0.005,
            "springLength": 250,
            "springConstant": 0.08,
            "damping": 0.8
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 250}
    },
    "interaction": {"hover": True, "tooltipDelay": 100}
}))

for node, data in G_cof.nodes(data=True):
    freq  = data['freq']
    short = (node.replace('Restaurante Universitário', 'RU')
                 .replace('Faculdade de ', 'Fac. ')
                 .replace(' (Amarelinho)', ''))
    size  = 9 + (freq / freq_max_g) * 45
    color = freq_color(freq)
    bc    = betweenness.get(node, 0)
    title = f"{node}\nMenções: {freq}\nBetweenness: {bc:.3f}"
    net.add_node(node, label=f"{short}\n({freq})", size=size,
                 color=color, title=title)

w_max = max((w for _, _, w in G_cof.edges(data='weight')), default=1)
for a, b, w in G_cof.edges(data='weight'):
    alpha = int(55 + (w / w_max) * 185)
    net.add_edge(a, b, value=w,
                 title=f'Co-frequência: {w}',
                 color=f'#{alpha:02x}{alpha:02x}{min(alpha+30, 255):02x}')

# Extrair apenas o conteúdo interno do <body> do HTML gerado pelo PyVis
pyvis_tmp = os.path.join(OUT, '_tmp_net.html')
net.save_graph(pyvis_tmp)
with open(pyvis_tmp, encoding='utf-8') as f:
    raw_pyvis = f.read()

# Extrair scripts do vis.js (CDN links) e o bloco de inicialização
vis_css_m  = re.search(r'(<link[^>]+vis-network[^>]+>)', raw_pyvis)
vis_js_m   = re.search(r'(<script[^>]+vis-network[^>]+></script>)', raw_pyvis)
vis_init_m = re.search(r'<script type="text/javascript">(.*?)</script>',
                        raw_pyvis, re.DOTALL)

vis_css    = vis_css_m.group(1)  if vis_css_m  else ''
vis_js     = vis_js_m.group(1)   if vis_js_m   else ''
vis_init   = vis_init_m.group(1) if vis_init_m else ''

print('  PyVis OK')

# ── 6. CORRELAÇÕES OTIMIZADAS (TOLERÂNCIA 100m) ─────────────────
print('Calculando correlações (Tolerância 100m)...')

axial_utm['_cpt'] = axial_utm.geometry.interpolate(0.5, normalized=True)
camadas_utm       = camadas_gdf.to_crs('EPSG:32723')

# OTIMIZAÇÃO: Cria buffers de 100m e usa spatial join (sjoin) indexado 
camadas_buffer = camadas_utm.copy()
camadas_buffer['geometry'] = camadas_buffer.geometry.buffer(100)

joined_spatial = gpd.sjoin(camadas_buffer, axial_utm, how='left', predicate='intersects')

results_corr = []
metric_keys = list(METRICS.keys())

for idx, group in joined_spatial.groupby(joined_spatial.index):
    row = camadas_utm.loc[idx]
    res = {'freq': row['freq'], 'Bloco_unb': row['Bloco_unb']}
    
    if pd.isna(group['index_right'].iloc[0]):
        nearest_idx = axial_utm.distance(row.geometry).argmin()
        best = axial_utm.iloc[nearest_idx]
    else:
        best = group[metric_keys].max(numeric_only=True)
        
    for m in metric_keys:
        res[m] = best.get(m, 0)
    results_corr.append(res)

joined = pd.DataFrame(results_corr)

correlations = {}
for col in METRICS:
    data = joined[['freq', col]].dropna()
    correlations[col] = data['freq'].corr(data[col]) if len(data) > 2 else 0.0


# ── 7. GRÁFICOS PLOTLY ───────────────────────────────────────────
n_resp          = len(df)
df['n_places']  = df[place_cols].notna().sum(axis=1)
modal_counts    = df['modal'].value_counts()
situacao_counts = df['situacao_unb'].value_counts()
top_places      = place_freq.most_common(25)
top_ra          = list(df['ra'].value_counts().head(10).items())
int_hh          = axial_gdf['INThh'].values
conn_v          = axial_gdf['CONN'].values

HOVER_STYLE = {"bgcolor": "#161b22", "bordercolor": "#4a90b8",
               "font": {"family": "IBM Plex Mono", "color": TEXT}}
BASE_LAYOUT = {
    "paper_bgcolor": "#161b22", "plot_bgcolor": "#0d1117",
    "font": {"color": MUTED, "family": "IBM Plex Mono", "size": 11},
    "hoverlabel": HOVER_STYLE,
    "margin": {"l": 10, "r": 55, "t": 38, "b": 20},
}

def j(obj):
    return json.dumps(obj, ensure_ascii=False)

def plotly_script(div_id, data_obj, layout_obj, extra=None):
    cfg = extra or {"responsive": True, "displayModeBar": False}
    return (f'<script>'
            f'(function(){{'
            f'var _d={j(data_obj)};'
            f'var _l={j(layout_obj)};'
            f'Plotly.newPlot("{div_id}",_d,_l,{j(cfg)});'
            f'}})();'
            f'</script>')

def hbar_chart(data, title, height=700):
    labels = [d[0][:48] + ('…' if len(d[0]) > 48 else '') for d in data]
    values = [d[1] for d in data]
    cols   = [f'rgba({int(74+i*4)},{int(144-i*2)},{int(184-i*2)},0.88)' for i in range(len(values))]
    layout = {**BASE_LAYOUT,
              "title": {"text": title, "font": {"size": 11, "color": TEXT, "family": "IBM Plex Mono"}},
              "xaxis": {"gridcolor": "#21262d", "zeroline": False},
              "yaxis": {"autorange": "reversed", "tickfont": {"size": 10}},
              "height": height, "margin": {"l": 280, "r": 65, "t": 38, "b": 16}}
    data_obj = [{"type": "bar", "orientation": "h", "x": values, "y": labels,
                 "text": [str(v) for v in values], "textposition": "outside",
                 "marker": {"color": cols}, "hovertemplate": "<b>%{y}</b><br>%{x}<extra></extra>"}]
    return data_obj, layout

def donut_chart(data, title, height=300):
    labels = [d[0] for d in data]
    values = [d[1] for d in data]
    COLS   = ['#4a90b8','#e6b84a','#5ec491','#b87a4a','#b84a6e','#7a4ab8','#4ab8a0','#a0b84a']
    layout = {**BASE_LAYOUT, "paper_bgcolor": "#161b22", "plot_bgcolor": "#161b22",
              "title": {"text": title, "font": {"size": 11, "color": TEXT, "family": "IBM Plex Mono"}},
              "legend": {"font": {"size": 9}, "bgcolor": "rgba(0,0,0,0)"},
              "height": height, "margin": {"l": 10, "r": 10, "t": 38, "b": 10}}
    data_obj = [{"type": "pie", "hole": 0.58, "labels": labels, "values": values,
                 "marker": {"colors": COLS[:len(labels)], "line": {"color": "#0d1117", "width": 1.5}},
                 "textinfo": "percent", "textfont": {"size": 10, "family": "IBM Plex Mono"},
                 "hovertemplate": "<b>%{label}</b><br>%{value} (%{percent})<extra></extra>", "sort": False}]
    return data_obj, layout

def hist_chart(values, title, xlabel, color='#4a90b8', height=240, n_bins=12):
    counts, edges = np.histogram(values, bins=n_bins)
    centers = [(edges[i]+edges[i+1])/2 for i in range(len(counts))]
    layout = {**BASE_LAYOUT,
              "title": {"text": title, "font": {"size": 11, "color": TEXT, "family": "IBM Plex Mono"}},
              "xaxis": {"title": xlabel, "gridcolor": "#21262d"},
              "yaxis": {"title": "Eixos",  "gridcolor": "#21262d"},
              "height": height, "margin": {"l": 10, "r": 10, "t": 38, "b": 60}, "bargap": 0.04}
    data_obj = [{"type": "bar", "x": [f'{c:.3f}' for c in centers], "y": counts.tolist(),
                 "marker": {"color": color, "opacity": 0.82, "line": {"color": "#0d1117", "width": 0.8}},
                 "hovertemplate": f"{xlabel}: %{{x}}<br>Eixos: %{{y}}<extra></extra>"}]
    return data_obj, layout

def scatter_chart(metric_col, label, height=270):
    data  = joined[['Bloco_unb', 'freq', metric_col]].dropna()
    corr  = correlations[metric_col]
    x = data[metric_col].values.astype(float)
    y = data['freq'].values.astype(float)
    m, bc = np.polyfit(x, y, 1) if len(x) > 1 else (0, 0)
    x_l   = [float(x.min()), float(x.max())]
    y_l   = [m * xv + bc for xv in x_l]
    layout = {**BASE_LAYOUT,
              "title": {"text": f"Dispersão: frequência × {label}  (r = {corr:+.3f})",
                        "font": {"size": 10, "color": TEXT, "family": "IBM Plex Mono"}},
              "xaxis": {"title": label, "gridcolor": "#21262d"},
              "yaxis": {"title": "Menções", "gridcolor": "#21262d"},
              "legend": {"font": {"size": 9}, "bgcolor": "rgba(0,0,0,0)"},
              "height": height, "margin": {"l": 10, "r": 10, "t": 38, "b": 60}}
    data_obj = [
        {"type": "scatter", "mode": "markers", "x": x.tolist(), "y": y.tolist(), "text": data['Bloco_unb'].tolist(),
         "marker": {"size": 9, "color": "#4a90b8", "opacity": 0.85, "line": {"color": "#e6edf3", "width": 0.5}},
         "hovertemplate": f"<b>%{{text}}</b><br>{label}: %{{x:.3f}}<br>Menções: %{{y}}<extra></extra>", "name": "Locais"},
        {"type": "scatter", "mode": "lines", "x": x_l, "y": y_l,
         "line": {"color": "#e6b84a", "width": 1.4, "dash": "dot"}, "hoverinfo": "skip", "name": f"r = {corr:+.3f}"}
    ]
    return data_obj, layout

d_places,   l_places   = hbar_chart(top_places, 'Top 25 locais frequentados', height=700)
d_ra,       l_ra       = hbar_chart(top_ra, 'Top 10 regiões administrativas', height=340)
d_modal,    l_modal    = donut_chart(list(modal_counts.items()),    'Modo de Transporte')
d_situacao, l_situacao = donut_chart(list(situacao_counts.items()), 'Vínculo com a UnB')
d_hist_int, l_hist_int = hist_chart(int_hh, 'Distribuição de Integração HH', 'Integração HH', '#4a90b8')
d_hist_con, l_hist_con = hist_chart(conn_v, 'Distribuição de Conectividade',  'Conectividade',  '#5ec491')


# ── 7.5 MAPAS ───────────────────────────────────────────────────
print('Gerando mapas interativos...')
camadas_wgs       = camadas_gdf.to_crs('EPSG:4326')
camadas_wgs['lat'] = camadas_wgs.geometry.y
camadas_wgs['lon'] = camadas_wgs.geometry.x
map_data = camadas_wgs[camadas_wgs['freq'] > 0].copy()

# Mapa de frequência de locais (Interativo)
fig_map = px.scatter_mapbox(
    map_data, lat='lat', lon='lon',
    hover_name='Bloco_unb',
    hover_data={'freq': True, 'lat': False, 'lon': False},
    size='freq', color='freq',
    color_continuous_scale='Turbo',
    zoom=14.5, mapbox_style='carto-darkmatter',
    title='Mapa de Frequência Interativo'
)
fig_map.update_layout(
    margin={'r': 0, 't': 40, 'l': 0, 'b': 0},
    paper_bgcolor='#161b22',
    font={'color': '#e6edf3', 'family': 'IBM Plex Mono'}
)
fig_map_json = json.loads(fig_map.to_json())
d_map = fig_map_json['data']
l_map = fig_map_json['layout']

# Mapa de rotas declaradas convertido para PNG estático para aliviar peso do HTML
print('Gerando PNG de rotas de circulação (Densidade)...')

coords_dict = {row['Bloco_unb']: (row['lat'], row['lon']) for _, row in camadas_wgs.iterrows()}
segments = []

for _, row in df.iterrows():
    places = [str(row[c]).strip() for c in ['p1','p2','p3','p4','p5']
              if pd.notna(row[c]) and str(row[c]).strip() != '']
    valid_points = [coords_dict[p] for p in places if p in coords_dict]
    if len(valid_points) > 1:
        for i in range(len(valid_points) - 1):
            p1 = (valid_points[i][1], valid_points[i][0]) # (lon, lat)
            p2 = (valid_points[i+1][1], valid_points[i+1][0])
            segments.append([p1, p2])

fig_routes, ax_routes = plt.subplots(figsize=(10, 8), facecolor=BG)
ax_routes.set_facecolor(BG)
ax_routes.set_aspect('equal')

if segments:
    # 1. Desenha as linhas de densidade
    lc = LineCollection(segments, color='#4a90b8', linewidths=2.5, alpha=0.03)
    ax_routes.add_collection(lc)
    
    all_lons = [pt[0] for seg in segments for pt in seg]
    all_lats = [pt[1] for seg in segments for pt in seg]
    min_lon, max_lon = min(all_lons), max(all_lons)
    min_lat, max_lat = min(all_lats), max(all_lats)
    
    dx = (max_lon - min_lon) * 0.05
    dy = (max_lat - min_lat) * 0.05
    ax_routes.set_xlim(min_lon - dx, max_lon + dx)
    ax_routes.set_ylim(min_lat - dy, max_lat + dy)
    
    # 2. Adiciona os pontos (scatter) proporcionalmente à frequência
    freq_pts = camadas_wgs[camadas_wgs['freq'] > 0]
    if not freq_pts.empty:
        # Tamanho dinâmico para os pontos
        sizes = 10 + (freq_pts['freq'] / freq_pts['freq'].max()) * 120
        ax_routes.scatter(freq_pts['lon'], freq_pts['lat'], s=sizes, 
                          color='#e6b84a', edgecolors='#161b22', 
                          linewidths=0.6, alpha=0.85, zorder=3)
    
    try:
        import contextily as cx
        cx.add_basemap(ax_routes, crs='EPSG:4326',
                       source=cx.providers.CartoDB.DarkMatterNoLabels,
                       alpha=0.6, zorder=0)
    except Exception:
        pass

ax_routes.set_axis_off()
ax_routes.set_title('Densidade de Rotas e Locais Declarados', color=TEXT, fontsize=12, 
                    fontweight='bold', fontfamily='monospace', pad=15)

routes_img_path = os.path.join(OUT, 'assets', 'maps', 'mapa_rotas.png')
fig_routes.savefig(routes_img_path, dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig_routes)


# ── 8. HELPERS HTML ──────────────────────────────────────────────
def corr_color(v):
    av = abs(v)
    if av >= 0.4:  return '#5ec491'
    if av >= 0.22: return '#e6b84a'
    return '#7d8590'

def corr_label(v):
    av, sign = abs(v), ('positiva' if v >= 0 else 'negativa')
    if av >= 0.5:  return f'Correlação {sign} moderada a forte'
    if av >= 0.3:  return f'Correlação {sign} fraca a moderada'
    if av >= 0.15: return f'Tendência {sign} fraca'
    return 'Correlação muito fraca'

def stat(label, val, sub=''):
    return (f'<div class="kpi"><div class="kpi-v">{val}</div>'
            f'<div class="kpi-l">{label}</div>'
            + (f'<div class="kpi-s">{sub}</div>' if sub else '') + '</div>')

def metric_block(col, info):
    label, inv, desc, val_desc, context = info
    corr = correlations[col]
    cc   = corr_color(corr)
    cl   = corr_label(corr)
    s    = axial_gdf[col]
    sc_data, sc_layout = scatter_chart(col, label)
    sc_script = plotly_script(f'sc_{col.lower()}', sc_data, sc_layout)
    inv_note  = '  ·  escala de cor invertida' if inv else ''
    return f'''
<div class="metric-block" id="metric-{col.lower()}">
  <div class="metric-header">
    <div>
      <div class="metric-name">{label}</div>
      <div class="metric-code">{col}{inv_note}</div>
    </div>
    <div class="metric-corr-box">
      <div class="metric-corr-val" style="color:{cc}">{corr:+.4f}</div>
      <div class="metric-corr-label">{cl}</div>
      <div class="metric-corr-label" style="font-size:0.6rem; color:#484f58;">(Tolerância 100m)</div>
    </div>
  </div>

  <div style="padding:15px 18px 0px;">
    <p style="font-size:0.85rem; color:#c9d1d9; margin-bottom:6px;"><strong>O que é?</strong> {desc}</p>
    <p style="font-size:0.85rem; color:#c9d1d9; margin-bottom:6px;"><strong>Valores:</strong> {val_desc}</p>
    <p style="font-size:0.85rem; color:#e6b84a; margin-bottom:12px;"><strong>Contexto (Dissertação):</strong> {context}</p>
  </div>

  <div class="metric-body">
    <div class="map-img-wrap">
      <img src="assets/maps/mapa_{col.lower()}.png" alt="Mapa {label}" class="map-img" loading="lazy">
    </div>
    <div class="metric-right">
      <div class="metric-stats">
        <div class="mstat"><span class="mstat-l">Mínimo</span><span class="mstat-v">{s.min():.3f}</span></div>
        <div class="mstat"><span class="mstat-l">Mediana</span><span class="mstat-v">{s.median():.3f}</span></div>
        <div class="mstat"><span class="mstat-l">Máximo</span><span class="mstat-v">{s.max():.3f}</span></div>
        <div class="mstat"><span class="mstat-l">Desvio-p.</span><span class="mstat-v">{s.std():.3f}</span></div>
      </div>
      <div id="sc_{col.lower()}" class="plotly-chart" style="flex:1;min-height:200px;padding:4px"></div>
    </div>
  </div>
</div>
{sc_script}'''

metrics_html = '\n'.join(metric_block(col, info) for col, info in METRICS.items())

# ── 9. SCRIPTS PLOTLY PARA O CORPO DO HTML ───────────────────────
sc_situacao = plotly_script('ch_situacao', d_situacao, l_situacao)
sc_modal    = plotly_script('ch_modal',    d_modal,    l_modal)
sc_ra       = plotly_script('ch_ra',       d_ra,       l_ra)
sc_places   = plotly_script('ch_places',   d_places,   l_places)
sc_hist_int = plotly_script('hist_int',    d_hist_int, l_hist_int)
sc_hist_con = plotly_script('hist_con',    d_hist_con, l_hist_con)
sc_map      = plotly_script('ch_map',      d_map,      l_map)


# ── 10. MONTAR HTML FINAL ────────────────────────────────────────
print('Compondo HTML...')

HTML = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Estruturas e Fluxos de Circulação no Campus Darcy Ribeiro (UnB)</title>
<link rel="icon" type="image/png" href="data:image/png;base64,{b64}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
{vis_css}
{vis_js}
<style>
:root {{
  --bg:    #0d1117; --panel: #161b22; --bord:  #21262d; --bord2: #30363d;
  --text:  #e6edf3; --muted: #7d8590; --dim:   #484f58; --acc:   #4a90b8;
  --mono:  'IBM Plex Mono', monospace;
  --sans:  'IBM Plex Sans', system-ui, sans-serif;
}}
*, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}
html {{ scroll-behavior:smooth; }}
body {{ background:var(--bg); color:var(--text); font-family:var(--sans);
        font-size:14px; line-height:1.7; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1400px; margin:0 auto; padding:0 28px 90px; }}

header {{ border-bottom:1px solid var(--bord); padding:50px 28px 34px;
          max-width:1400px; margin:0 auto; }}
.eyebrow {{ font-family:var(--mono); font-size:0.68rem; color:var(--acc);
             letter-spacing:0.16em; text-transform:uppercase; margin-bottom:12px; }}
header h1 {{ font-family:var(--mono); font-size:2.8rem; font-weight:600;
              color:var(--text); line-height:1.2; margin-bottom:20px; }}
.badge-row {{ margin-top:16px; display:flex; flex-wrap:wrap; gap:5px; }}
.badge {{ font-family:var(--mono); font-size:0.66rem; color:var(--muted);
          border:1px solid var(--bord2); border-radius:3px; padding:2px 8px; }}

nav {{ position:sticky; top:0; z-index:100;
       background:rgba(13,17,23,0.94); backdrop-filter:blur(8px);
       border-bottom:1px solid var(--bord); padding:0 28px; }}
.nav-inner {{ max-width:1200px; margin:0 auto; display:flex; overflow-x:auto; }}
nav a {{ font-family:var(--mono); font-size:0.68rem; color:var(--muted);
         text-decoration:none; padding:11px 15px; letter-spacing:0.06em;
         text-transform:uppercase; border-bottom:2px solid transparent;
         white-space:nowrap; transition:color .16s, border-color .16s; }}
nav a:hover {{ color:var(--text); border-color:var(--acc); }}

section {{ padding-top:62px; }}
.sec-label {{ font-family:var(--mono); font-size:0.66rem; color:var(--acc);
              letter-spacing:0.16em; text-transform:uppercase; margin-bottom:5px; }}
.sec-title {{ font-family:var(--mono); font-size:0.98rem; font-weight:600;
              color:var(--text); margin-bottom:20px; padding-bottom:12px;
              border-bottom:1px solid var(--bord); }}

.kpi-grid {{ display:grid; grid-template-columns:1fr 1fr;
             gap:1px; background:var(--bord); border:1px solid var(--bord);
             border-radius:5px; overflow:hidden; margin-bottom:28px; }}
.kpi {{ background:var(--panel); padding:17px 13px; }}
.kpi-v {{ font-family:var(--mono); font-size:1.6rem; font-weight:600;
           color:var(--acc); line-height:1; }}
.kpi-l {{ font-size:0.68rem; color:var(--muted); text-transform:uppercase;
           letter-spacing:0.06em; margin-top:5px; }}
.kpi-s {{ font-size:0.65rem; color:var(--dim); margin-top:2px; font-family:var(--mono); }}

.panel {{ background:var(--panel); border:1px solid var(--bord); border-radius:5px; padding:18px; }}
.panel-title {{ font-family:var(--mono); font-size:0.67rem; color:var(--muted);
                text-transform:uppercase; letter-spacing:0.08em; margin-bottom:12px; }}
.plotly-chart {{ width:100%; }}

.callout {{ border-left:2px solid var(--bord2); padding:10px 14px; margin-bottom:10px;
            font-size:0.83rem; color:#9ca3af; line-height:1.65; }}
.callout strong {{ color:#c9d1d9; font-weight:500; }}

.intro-box {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:20px; margin-bottom:30px; }}
.intro-box h3 {{ font-size:1.0rem; color:#4a90b8; margin-bottom:10px; font-family:var(--mono); }}

.two-col {{ display:grid; grid-template-columns:1.1fr 0.9fr; gap:22px; align-items:start; }}
.three-col {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:18px; }}
.two-eq  {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}

.storymap-container {{ display:grid; grid-template-columns:350px 1fr; gap:40px; align-items:start; margin-top:40px; }}
.story-sidebar {{ position:sticky; top:80px; display:flex; flex-direction:column; gap:20px; }}

table {{ width:100%; border-collapse:collapse; font-size:0.8rem; }}
thead th {{ font-family:var(--mono); font-size:0.66rem; text-transform:uppercase;
            letter-spacing:0.07em; color:var(--muted); padding:7px 11px;
            border-bottom:1px solid var(--bord); text-align:left; font-weight:500; }}
tbody td {{ padding:6px 11px; border-bottom:1px solid var(--bord); color:var(--text); }}
td.r {{ text-align:right; font-family:var(--mono); color:var(--muted); }}
tbody tr:hover td {{ background:rgba(255,255,255,0.025); }}

/* PYVIS */
.net-wrap {{ background:#0d1117; border:1px solid var(--bord); border-radius:5px;
             overflow:hidden; position:relative; margin-bottom:20px; }}
#mynetwork {{ width:100%; height:600px; }}
.net-legend {{ position:absolute; bottom:12px; left:12px;
               background:rgba(13,17,23,0.88); border:1px solid var(--bord);
               border-radius:4px; padding:8px 12px;
               font-family:var(--mono); font-size:0.66rem; color:var(--muted);
               z-index:10; pointer-events:none; }}
.net-leg-row {{ display:flex; align-items:center; gap:6px; margin-bottom:3px; }}
.net-leg-dot {{ width:9px; height:9px; border-radius:50%; flex-shrink:0; }}

/* METRIC BLOCKS */
.metric-block {{ background:var(--panel); border:1px solid var(--bord);
                 border-radius:5px; overflow:hidden; margin-bottom:16px; }}
.metric-header {{ display:flex; justify-content:space-between; align-items:flex-start;
                  padding:15px 18px 12px; border-bottom:1px solid var(--bord); gap:16px; }}
.metric-name {{ font-family:var(--mono); font-size:0.9rem; font-weight:600; color:var(--text); }}
.metric-code {{ font-family:var(--mono); font-size:0.66rem; color:var(--dim); margin-top:3px; }}
.metric-corr-box {{ text-align:right; flex-shrink:0; }}
.metric-corr-val {{ font-family:var(--mono); font-size:1.4rem; font-weight:600; line-height:1; }}
.metric-corr-label {{ font-size:0.68rem; color:var(--muted); margin-top:3px; }}
.metric-body {{ display:grid; grid-template-columns:1fr 1fr; border-top:1px solid var(--bord); gap:20px; padding-top:15px; }}
.map-img-wrap {{ background:#0d1117; padding:8px; border-right:1px solid var(--bord);
                 display:flex; align-items:center; justify-content:center;
                 width:100%; height:100%; box-sizing:border-box; }}
.map-img {{ width:100%; display:block; border-radius:3px; }}
.metric-right {{ display:flex; flex-direction:column; }}
.metric-stats {{ display:grid; grid-template-columns:1fr 1fr; border-bottom:1px solid var(--bord); }}
.mstat {{ display:flex; justify-content:space-between; align-items:baseline;
          padding:7px 13px; border-right:1px solid var(--bord);
          border-bottom:1px solid var(--bord); gap:8px; }}
.mstat:nth-child(even) {{ border-right:none; }}
.mstat:nth-last-child(-n+2) {{ border-bottom:none; }}
.mstat-l {{ font-size:0.7rem; color:var(--muted); }}
.mstat-v {{ font-family:var(--mono); font-size:0.8rem; color:var(--text); }}

@media(max-width:960px) {{
  .three-col, .two-col, .two-eq, .storymap-container {{ grid-template-columns:1fr; }}
  .story-sidebar {{ position:relative; top:0; }}
  .metric-body {{ grid-template-columns:1fr; }}
  .map-img-wrap {{ width:100%; border-right:none; border-bottom:1px solid var(--bord); }}
  .metric-right {{ border-left:none; }}
}}
</style>
</head>
<body>

<header>
  <h1>Estruturas e Fluxos de Circulação<br>no Campus Darcy Ribeiro:</h1>
      <div class="eyebrow">potencialidades da sintaxe espacial aliada a técnicas de geoprocessamento</div>
  <div class="badge-row">
    <span class="badge">city2graph</span>
    <span class="badge">NetworkX</span>
    <span class="badge">GeoPandas</span>
    <span class="badge">PyVis 0.3.2</span>
    <span class="badge">Plotly 2.32</span>
  </div>
</header>

<nav>
  <div class="nav-inner">
    <a href="#fluxos">01 · Dinâmica de Fluxos</a>
    <a href="#perfil">02 · Composição</a>
    <a href="#locais">03 · Locais</a>
    <a href="#metricas">04 · Métricas Axiais</a>
  </div>
</nav>

<div class="wrap">

<div class="storymap-container">
  <aside class="story-sidebar">
    <div class="intro-box">
      <h3>O Modelo Axial e a Sintaxe Espacial</h3>
      <p style="margin-bottom:12px; font-size:0.88rem; color:#9ca3af; line-height:1.6;">A <strong>Sintaxe Espacial</strong>, concebida por Hillier e Hanson (1984), é uma teoria que integra sociologia e arquitetura através de representações matemáticas do espaço. Nas análises suportadas por essa teoria, os eixos das vias são simplificados em <strong>grafos</strong>, a partir dos quais são traçadas diversas métricas. A Sintaxe Espacial está preocupada com a <strong>forma</strong> do espaço analisado, não necessariamente com seu conteúdo, isto é, os atratores de um determinado local a ser frequentado.</p>
      <h3>A Dissertação (Ferraz Junior, 2025)</h3>
      <p style="font-size:0.88rem; color:#9ca3af; line-height:1.6;">Este site explora resultados da pesquisa de Mestrado intitulada "Estruturas e Fluxos de Circulação no Campus Darcy Ribeiro: potencialidades da sintaxe espacial aliada a técnicas de geoprocessamento", defendida e publicada em 2025. O estudo revelou que, no Campus Darcy Ribeiro, a integração topológica nem sempre se alinha perfeitamente ao uso efetivo. Por exemplo, a via L3 possui alta integração matemática, mas não é a mais frequentada, sugerindo que os <strong>atratores locais</strong> exercem forte influência. Mesmo assim, a teoria pode fornecer insights valiosos para o planejamento da mobilidade no Campus.</p>
    </div>

    <div class="kpi-grid">
      {stat("Respondentes", n_resp)}
      {stat("Locais únicos", len(place_freq))}
      {stat("Locais/pessoa", f"{df['n_places'].mean():.1f}")}
      {stat("Dias/semana", f"{df['dias_freq'].mean():.1f}", f"dp = {df['dias_freq'].std():.1f}")}
      {stat("Eixos axiais", n_edges, f"{n_nodes} nós")}
      {stat("Integ. HH", f"{np.mean(int_hh):.3f}", f"máx={np.max(int_hh):.3f}")}
    </div>

    <div class="intro-box" style="border-left: 2px solid #e6b84a;">
      <p style="font-size: 0.8rem; color: #7d8590;">
        <strong>Nota Metodológica:</strong> Dados coletados via formulário estruturado no 1º semestre de 2024. Pesquisa de Mestrado (2025) financiada pela CAPES.
      </p>
    </div>
  </aside>

  <div class="story-main">
    <section id="fluxos" style="padding-top:0;">
      <div class="sec-label">01 · Dinâmica de Fluxos</div>
      <div class="sec-title">Redes e Distribuição de Frequência</div>
      <div class="callout" style="margin-bottom:20px">
        Abaixo, a exploração visual da rotina acadêmica. O <strong>Mapa Interativo</strong> mostra a intensidade do uso declarado, e o mapa de densidade de rotas ilustra as densidades de conexão entre locais. Por fim, a <strong>Rede de Co-frequência</strong> revela como os usuários vinculam social e funcionalmente esses mesmos espaços.
      </div>

      <div class="panel" style="padding:0; overflow:hidden; margin-top:20px;">
        <div id="ch_map" style="width:100%; height:750px;"></div>
      </div>

      <div class="panel" style="padding:20px; margin-top:20px; display:flex; justify-content:center; background:#0d1117;">
        <img src="assets/maps/mapa_rotas.png" alt="Densidade de Rotas" style="max-width:100%; border-radius:6px; border: 1px solid var(--bord);">
      </div>

      <div class="net-wrap" style="margin-top:20px;">
        <div id="mynetwork"></div>
        <div class="net-legend">
          <div class="net-leg-row"><div class="net-leg-dot" style="background:#3c5a82"></div><span>menor frequência</span></div>
          <div class="net-leg-row"><div class="net-leg-dot" style="background:#dc6400"></div><span>maior frequência</span></div>
          <div style="margin-top:5px;border-top:1px solid #21262d;padding-top:5px">Espessura = co-frequência</div>
        </div>
      </div>
    </section>

    <section id="perfil">
      <div class="sec-label">02 · Perfil dos Respondentes</div>
      <div class="sec-title">Composição da amostra</div>
      <div class="two-eq">
        <div class="panel">
          <div class="panel-title">Vínculo com a UnB</div>
          <div id="ch_situacao" class="plotly-chart"></div>
        </div>
        <div class="panel">
          <div class="panel-title">Modo de Transporte</div>
          <div id="ch_modal" class="plotly-chart"></div>
        </div>
      </div>
      <div class="panel" style="margin-top: 18px;">
        <div class="panel-title">Origem (Região Administrativa)</div>
        <div id="ch_ra" class="plotly-chart"></div>
      </div>
    </section>

    <section id="locais">
      <div class="sec-label">03 · Locais Frequentados</div>
      <div class="sec-title">Uso Absoluto Declarado</div>
      <div class="panel" style="margin-bottom:20px">
        <div id="ch_places" class="plotly-chart"></div>
      </div>
    </section>

    <section id="metricas">
      <div class="sec-label">04 · Métricas de Configuração Espacial</div>
      <div class="sec-title">Modelo axial × Frequência declarada</div>
      <div class="two-eq" style="margin-bottom:20px">
        <div class="panel">
          <div class="panel-title">Distribuição: Integração HH</div>
          <div id="hist_int" class="plotly-chart"></div>
        </div>
        <div class="panel">
          <div class="panel-title">Distribuição: Conectividade</div>
          <div id="hist_con" class="plotly-chart"></div>
        </div>
      </div>
      {metrics_html}
    </section>
  </div>
</div>

</div>

<footer style="line-height:1.8; color:var(--muted); text-align:center; border-top:1px solid var(--bord); margin-top:80px; padding:40px 20px;">
  <em>ELaborado em 2026</em><br><br>
  <span style="font-size:0.75rem; color:var(--dim);">
    Gerado com Python · city2graph {city2graph.__version__} · Plotly · PyVis · NetworkX
  </span>
</footer>

{sc_map}
{sc_situacao}
{sc_modal}
{sc_ra}
{sc_places}
{sc_hist_int}
{sc_hist_con}

<script type="text/javascript">
{vis_init}
</script>

</body>
</html>"""

out_html = os.path.join(OUT, 'index.html')
with open(out_html, 'w', encoding='utf-8') as f:
    f.write(HTML)

print(f'\n✅ Relatório gerado em: {out_html}')
print(f'   Mapas PNG em: {OUT}/assets/maps/')