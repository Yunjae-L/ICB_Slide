import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import koreanize_matplotlib as km

OUTPUT_DIR = "analysis_output"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
MARKDOWN_PATH = os.path.join(OUTPUT_DIR, "penguins_analysis.md")

os.makedirs(IMAGES_DIR, exist_ok=True)

# Load dataset
penguins = sns.load_dataset('penguins')
# Drop rows with missing values for simplicity
penguins = penguins.dropna().reset_index(drop=True)

# Basic info
info_text = penguins.describe(include='all').to_string()

# Prepare plots
images = []
km.koreanize()

# 1. Histogram: flipper_length_mm
fig, ax = plt.subplots()
ax.hist(penguins['flipper_length_mm'], bins=20, color='teal')
ax.set_title('지느러미 길이 (mm)')
img1 = os.path.join(IMAGES_DIR, 'hist_flipper_length.png')
fig.savefig(img1, bbox_inches='tight')
plt.close(fig)
images.append(('히스토그램: 지느러미 길이', img1))

# 2. Histogram: body_mass_g
fig, ax = plt.subplots()
ax.hist(penguins['body_mass_g'], bins=20, color='orange')
ax.set_title('체중 (g)')
img2 = os.path.join(IMAGES_DIR, 'hist_body_mass.png')
fig.savefig(img2, bbox_inches='tight')
plt.close(fig)
images.append(('히스토그램: 체중', img2))

# 3. Scatter: bill_length_mm vs bill_depth_mm (colored by species)
fig, ax = plt.subplots()
sns.scatterplot(data=penguins, x='bill_length_mm', y='bill_depth_mm', hue='species', ax=ax)
ax.set_title('부리 길이 vs 부리 깊이')
img3 = os.path.join(IMAGES_DIR, 'scatter_bill.png')
fig.savefig(img3, bbox_inches='tight')
plt.close(fig)
images.append(('산점도: 부리 길이 vs 부리 깊이', img3))

# 4. Pairplot (pairwise relationships)
pairplot_df = penguins.copy()
if 'year' in pairplot_df.columns:
    pairplot_df = pairplot_df.drop(columns=['year'])
pair = sns.pairplot(pairplot_df, hue='species')
img4 = os.path.join(IMAGES_DIR, 'pairplot.png')
pair.savefig(img4)
plt.close('all')
images.append(('변수들 간 관계 (Pairplot)', img4))

# 5. Boxplot: body_mass_g by species
fig, ax = plt.subplots()
sns.boxplot(data=penguins, x='species', y='body_mass_g', ax=ax)
ax.set_title('종별 체중 분포')
img5 = os.path.join(IMAGES_DIR, 'box_body_mass_by_species.png')
fig.savefig(img5, bbox_inches='tight')
plt.close(fig)
images.append(('상자그림: 종별 체중 분포', img5))

# 6. Violinplot: flipper_length_mm by species
fig, ax = plt.subplots()
sns.violinplot(data=penguins, x='species', y='flipper_length_mm', ax=ax)
ax.set_title('종별 지느러미 길이 분포')
img6 = os.path.join(IMAGES_DIR, 'violin_flipper_by_species.png')
fig.savefig(img6, bbox_inches='tight')
plt.close(fig)
images.append(('바이올린플롯: 종별 지느러미 길이 분포', img6))

# 7. Countplot (bar): species counts
fig, ax = plt.subplots()
sns.countplot(data=penguins, x='species', palette='pastel', ax=ax)
ax.set_title('종별 개체수')
img7 = os.path.join(IMAGES_DIR, 'count_species.png')
fig.savefig(img7, bbox_inches='tight')
plt.close(fig)
images.append(('막대(갯수): 종별 개체수', img7))

# 8. Barplot: mean body_mass_g by species
fig, ax = plt.subplots()
sns.barplot(data=penguins, x='species', y='body_mass_g', ci='sd', palette='muted', ax=ax)
ax.set_title('종별 평균 체중')
img8 = os.path.join(IMAGES_DIR, 'bar_mean_body_mass_by_species.png')
fig.savefig(img8, bbox_inches='tight')
plt.close(fig)
images.append(('막대: 종별 평균 체중', img8))

# 9. Stacked bar: island vs species counts (using pivot and pandas plot)
ct_island_species = pd.crosstab(penguins['island'], penguins['species'])
fig = ct_island_species.plot(kind='bar', stacked=True, figsize=(6,4)).get_figure()
img9 = os.path.join(IMAGES_DIR, 'stacked_island_species.png')
fig.savefig(img9, bbox_inches='tight')
plt.close('all')
images.append(('누적막대: 섬별 종 개수', img9))

# 10. Heatmap: correlation matrix
corr = penguins.select_dtypes(include='number').corr()
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
ax.set_title('상관계수 행렬')
img10 = os.path.join(IMAGES_DIR, 'heatmap_corr.png')
fig.savefig(img10, bbox_inches='tight')
plt.close(fig)
images.append(('히트맵: 상관계수 행렬', img10))

# 11. Swarmplot: bill_length_mm by species
fig, ax = plt.subplots()
sns.swarmplot(data=penguins, x='species', y='bill_length_mm', hue='sex', dodge=True, ax=ax)
ax.set_title('종별/성별 부리 길이')
img11 = os.path.join(IMAGES_DIR, 'swarm_bill_length_species_sex.png')
fig.savefig(img11, bbox_inches='tight')
plt.close(fig)
images.append(('스웜플롯: 종별/성별 부리 길이', img11))

# 12. KDE: body_mass_g by species
fig, ax = plt.subplots()
for s in penguins['species'].unique():
    subset = penguins[penguins['species'] == s]
    sns.kdeplot(subset['body_mass_g'], label=s, ax=ax, fill=True)
ax.set_title('종별 체중 KDE')
ax.legend()
img12 = os.path.join(IMAGES_DIR, 'kde_body_mass_by_species.png')
fig.savefig(img12, bbox_inches='tight')
plt.close(fig)
images.append(('KDE: 종별 체중', img12))

# Additional plots if needed (make sure at least 10 exist)

# Prepare cross-tab and pivot outputs for bar graphs
# For countplot (species counts) -> crosstab species vs island
crosstab_species_island = pd.crosstab(penguins['species'], penguins['island'])
# For mean body_mass barplot -> pivot table species x sex mean body_mass
pivot_bodymass = penguins.pivot_table(index='species', columns='sex', values='body_mass_g', aggfunc='mean')

# Write markdown
with open(MARKDOWN_PATH, 'w', encoding='utf-8') as f:
    f.write('# Penguins 데이터셋 분석\n\n')
    f.write('데이터셋 요약 (결측값 제거 후):\n\n')
    f.write('```\n')
    f.write(info_text + '\n')
    f.write('```\n\n')

    f.write('## 생성된 그래프\n')
    for title, path in images:
        rel = os.path.relpath(path, OUTPUT_DIR)
        f.write(f'### {title}\n')
        f.write(f'![]({{}})\n\n'.format(os.path.join('images', os.path.basename(path))))

    f.write('## 막대그래프 관련 교차표 및 피봇테이블\n\n')
    f.write('### 종(species) vs 섬(island) 교차표\n\n')
    f.write(crosstab_species_island.to_markdown() + '\n\n')

    f.write('### 종(species) x 성별(sex) 평균 body_mass_pivot\n\n')
    f.write(pivot_bodymass.to_markdown() + '\n\n')

print('출력 경로:', OUTPUT_DIR)
print('마크다운 파일:', MARKDOWN_PATH)
print('생성된 이미지들:')
for _, p in images:
    print(p)

# Also save the crosstab and pivot as CSV for quick inspection
crosstab_species_island.to_csv(os.path.join(OUTPUT_DIR, 'crosstab_species_island.csv'))
pivot_bodymass.to_csv(os.path.join(OUTPUT_DIR, 'pivot_bodymass_species_sex.csv'))

print('완료')
