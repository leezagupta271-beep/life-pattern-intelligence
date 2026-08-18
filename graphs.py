import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey
import numpy as np
import pandas as pd
from scipy import stats

# Apply modern dark mode theme for matplotlib
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#2A1748'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#FFFFFF'
plt.rcParams['grid.alpha'] = 0.08
plt.rcParams['grid.linestyle'] = '--'

BG_COLOR = '#0D0B18'
CARD_BG = '#14102A'
TEXT_COLOR = '#FFFFFF'
MUTED_TEXT = '#94A3B8'
CYAN = '#00F5D4'
PURPLE = '#A855F7'
PINK = '#FF4D9D'
GREEN = '#22C55E'
ORANGE = '#F59E0B'
RED = '#FF5F6D'


def _fig_to_base64(fig):
    """Utility to convert a Matplotlib Figure to Base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=False, dpi=140)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_b64}"


def generate_productivity_heatmap(df, user_data=None):
    """1. Productivity Heatmap across days & time windows."""
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)

    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    hours = ['Morning', 'Noon', 'Afternoon', 'Evening', 'Night']

    if user_data is None:
        user_data = {}

    sleep = float(user_data.get('sleep', 7.5))
    water = float(user_data.get('water', 2.0))
    stress = float(user_data.get('stress', 4))
    study = float(user_data.get('study', 3.5))
    steps = float(user_data.get('steps', 8432))
    exercise = user_data.get('exercise', 'No')
    exercise_factor = 5 if str(exercise).lower() in ['yes', '1', 'true'] else -3

    np.random.seed(42)
    heatmap_data = np.random.uniform(60, 98, size=(len(hours), len(days)))
    heatmap_data[0, 1] = 95.0  # Tuesday morning peak
    heatmap_data[1, 1] = 92.5
    heatmap_data[-1, -1] = 62.0  # Sunday night low

    adjustment = (
        (sleep - 7.0) * 2.5
        + (water - 2.0) * 4.0
        - (stress - 4.0) * 3.5
        + (study - 3.5) * 2.0
        + (steps - 8432) / 1800.0
        + exercise_factor
    )
    adjustment = np.clip(adjustment, -14, 14)
    heatmap_data = np.clip(heatmap_data + adjustment, 50, 100)

    cax = ax.imshow(heatmap_data, cmap='magma', aspect='auto', vmin=50, vmax=100)
    cbar = fig.colorbar(cax, ax=ax)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)
    cbar.set_label('Productivity Intensity (%)', color=TEXT_COLOR, fontsize=9)

    ax.set_xticks(np.arange(len(days)))
    ax.set_yticks(np.arange(len(hours)))
    ax.set_xticklabels(days, color=TEXT_COLOR, fontsize=9)
    ax.set_yticklabels(hours, color=TEXT_COLOR, fontsize=9)

    for i in range(len(hours)):
        for j in range(len(days)):
            ax.text(j, i, f"{int(heatmap_data[i, j])}%", ha='center', va='center',
                    color='#FFFFFF' if heatmap_data[i, j] < 80 else '#000000', fontsize=8, fontweight='bold')

    ax.set_title('Weekly Productivity Heatmap', color=TEXT_COLOR, fontsize=12, fontweight='bold', pad=12)
    return _fig_to_base64(fig)


def generate_burnout_gauge(risk_score, risk_category):
    """2. Burnout Risk Gauge Arc."""
    fig, ax = plt.subplots(figsize=(6, 3.2), subplot_kw={'projection': 'polar'}, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Angle mapping for semi-circle gauge (180 deg to 0 deg)
    theta = np.linspace(np.pi, 0, 100)
    r = np.ones_like(theta)

    # Plot arc background
    ax.plot(theta, r, color='#2A1748', linewidth=16)

    # Color mapping
    if risk_score < 35:
        color = GREEN
    elif risk_score < 65:
        color = ORANGE
    else:
        color = RED

    fill_angle = np.pi * (1.0 - (risk_score / 100.0))
    fill_theta = np.linspace(np.pi, fill_angle, 100)
    fill_r = np.ones_like(fill_theta)

    ax.plot(fill_theta, fill_r, color=color, linewidth=16)

    # Needle
    needle_angle = np.pi * (1.0 - (risk_score / 100.0))
    ax.annotate('', xy=(needle_angle, 0.9), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=TEXT_COLOR, lw=2.5))

    ax.set_ylim(0, 1.15)
    ax.axis('off')

    ax.text(0, 0, f"{risk_score}%\n{risk_category.upper()}", ha='center', va='center',
            color=color, fontsize=14, fontweight='bold')

    fig.suptitle('Burnout Risk Gauge', color=TEXT_COLOR, fontsize=12, fontweight='bold', y=0.98)
    return _fig_to_base64(fig)


def generate_habit_impact_chart(correlations):
    """3. Habit Impact Horizontal Bar Chart."""
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)

    habits = list(correlations.keys())
    values = list(correlations.values())

    colors = [CYAN if v >= 0 else RED for v in values]

    y_pos = np.arange(len(habits))
    bars = ax.barh(y_pos, values, color=colors, height=0.65, edgecolor='none')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(habits, color=TEXT_COLOR, fontsize=9)
    ax.set_xlabel('Pearson Correlation with Productivity', color=TEXT_COLOR, fontsize=9)
    ax.set_xlim(-0.6, 0.8)

    ax.axvline(0, color=MUTED_TEXT, linestyle='--', linewidth=0.8)

    for bar, val in zip(bars, values):
        width = bar.get_width()
        offset = 0.02 if width >= 0 else -0.07
        ax.text(width + offset, bar.get_y() + bar.get_height()/2.0, f"{val:+.2f}",
                va='center', color=TEXT_COLOR, fontsize=8, fontweight='bold')

    ax.tick_params(colors=TEXT_COLOR)
    ax.set_title('Habit Impact Analysis (Correlation)', color=TEXT_COLOR, fontsize=12, fontweight='bold', pad=10)
    return _fig_to_base64(fig)


def generate_sleep_trend_chart(user_df, user_sleep=7.5):
    """4. Sleep Analytics Trend Chart."""
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)

    if user_df is not None and not user_df.empty and 'sleep' in user_df.columns:
        sleep_data = pd.to_numeric(user_df['sleep'].tail(14), errors='coerce').fillna(0).values
        dates = user_df['date'].tail(14).values
        days = [str(d).split('-')[-1] for d in dates]
    else:
        np.random.seed(12)
        days = [f"Day {i+1}" for i in range(14)]
        sleep_data = np.random.normal(user_sleep, 0.6, 14)
        sleep_data[-1] = user_sleep

    moving_avg = pd.Series(sleep_data).rolling(window=3, min_periods=1).mean()

    ax.plot(days, sleep_data, marker='o', color=PURPLE, linewidth=2, label='Daily Sleep (h)')
    ax.plot(days, moving_avg, linestyle='--', color=CYAN, linewidth=2, label='3-Day Moving Avg')

    ax.axhline(8.0, color=GREEN, linestyle=':', linewidth=1.5, label='Optimal Target (8h)')

    ax.set_ylabel('Sleep Hours', color=TEXT_COLOR, fontsize=9)
    ax.set_xticklabels(days, rotation=45, color=TEXT_COLOR, fontsize=8)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, alpha=0.15)
    ax.legend(facecolor=CARD_BG, edgecolor=MUTED_TEXT, labelcolor=TEXT_COLOR, fontsize=8)

    ax.set_title('Sleep Analytics & Trend', color=TEXT_COLOR, fontsize=12, fontweight='bold', pad=10)
    return _fig_to_base64(fig)


def generate_radar_chart(user_data, radar_data=None):
    """6. Polar Radar Chart — fully dynamic from analytics radar_data dict."""
    if radar_data is None:
        # Build a minimal radar_data inline so the chart still works standalone
        import analytics as _ana
        radar_data = _ana.calculate_radar_scores(user_data, prediction=user_data.get('prediction', 85.0))

    labels     = radar_data['axes']
    user_vals  = radar_data['user_scores']
    bench_vals = radar_data['benchmark']
    num_vars   = len(labels)

    # Close the polygon
    user_vals_plot  = user_vals  + user_vals[:1]
    bench_vals_plot = bench_vals + bench_vals[:1]

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles_plot = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={'projection': 'polar'}, facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)

    # Grid rings
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], color=MUTED_TEXT, fontsize=7.5)
    ax.yaxis.set_tick_params(pad=3)

    # Subtle concentric ring fills
    for ring, alpha in [(100, 0.04), (80, 0.03), (60, 0.02)]:
        ax.fill(angles_plot, [ring] * (num_vars + 1), color='white', alpha=alpha)

    # Benchmark polygon
    ax.plot(angles_plot, bench_vals_plot, color=PURPLE, linestyle='--', linewidth=1.8,
            label='Top Quartile Benchmark', alpha=0.85)
    ax.fill(angles_plot, bench_vals_plot, color=PURPLE, alpha=0.10)

    # User polygon
    ax.plot(angles_plot, user_vals_plot, color=CYAN, linewidth=2.5, label='Your Scores')
    ax.fill(angles_plot, user_vals_plot, color=CYAN, alpha=0.22)

    # Vertex dots + value annotations
    for angle, val in zip(angles, user_vals):
        ax.scatter([angle], [val], color=CYAN, s=50, zorder=5, edgecolors='white', linewidths=0.8)
        # Push label outward past the polygon
        label_r = min(val + 12, 95)
        ax.text(angle, label_r, f"{int(val)}", ha='center', va='center',
                color=TEXT_COLOR, fontsize=8, fontweight='bold')

    # Axis labels — pad them out past the max ring
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    for angle, lbl in zip(angles, labels):
        # Break long labels
        short = lbl.replace(' & ', '\n& ')
        ax.text(angle, 113, short, ha='center', va='center',
                color=TEXT_COLOR, fontsize=8.5, fontweight='600',
                multialignment='center')

    ax.spines['polar'].set_color(MUTED_TEXT)
    ax.spines['polar'].set_alpha(0.3)
    ax.grid(color='white', alpha=0.10, linestyle='--', linewidth=0.6)

    ax.legend(
        loc='upper right',
        bbox_to_anchor=(1.38, 1.18),
        facecolor=CARD_BG,
        edgecolor=MUTED_TEXT,
        labelcolor=TEXT_COLOR,
        fontsize=8.5,
        framealpha=0.9,
    )

    ax.set_title('Life Balance Radar', color=TEXT_COLOR, fontsize=12, fontweight='bold', pad=28)
    fig.tight_layout(pad=1.5)
    return _fig_to_base64(fig)


def generate_cluster_chart(raw_df, user_data, cluster_data=None):
    """Lifestyle Cluster Scatter Plot — K-Means colored, Sleep vs Productivity."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler as _MMS

    CLUSTER_COLORS = [CYAN, PURPLE, RED, ORANGE]
    CLUSTER_NAMES  = ['Balanced Routine', 'High Performer', 'Recovery Mode', 'Stressed Overachiever']

    user_sleep = float(user_data.get('sleep', 7.5))
    user_prod  = float(user_data.get('prediction', 85.0))

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)

    plotted_clusters = set()

    if raw_df is not None and isinstance(raw_df, pd.DataFrame) and not raw_df.empty and \
       'Sleep' in raw_df.columns and 'Productivity' in raw_df.columns:
        try:
            df_plot = raw_df[['Sleep', 'Productivity', 'Stress', 'StudyHours']].dropna().sample(
                min(len(raw_df), 3000), random_state=42
            )
            X_plot = df_plot[['Sleep', 'Productivity', 'Stress', 'StudyHours']].values
            scaler_p = _MMS()
            X_scaled = scaler_p.fit_transform(X_plot)
            km = KMeans(n_clusters=4, random_state=42, n_init=10)
            km_labels = km.fit_predict(X_scaled)

            # Map KMeans labels → semantic order by Productivity centroid
            centroids_orig = scaler_p.inverse_transform(km.cluster_centers_)
            prod_col = 1  # index of Productivity
            prod_ranks = np.argsort(centroids_orig[:, prod_col])[::-1]  # high→low

            semantic_map = {}
            stress_col = 2
            study_col  = 3
            for rank, orig_idx in enumerate(prod_ranks):
                sc = centroids_orig[orig_idx, stress_col]
                stc = centroids_orig[orig_idx, study_col]
                if rank == 0:
                    semantic_map[orig_idx] = 1  # High Performer
                elif sc >= 6.5 and stc >= 4.0:
                    semantic_map[orig_idx] = 3  # Stressed Overachiever
                elif centroids_orig[orig_idx, 0] < 6.5 or sc >= 7.0:
                    semantic_map[orig_idx] = 2  # Recovery Mode
                else:
                    semantic_map[orig_idx] = 0  # Balanced Routine

            semantic_labels = np.array([semantic_map[l] for l in km_labels])

            # Plot each cluster
            for sem_id, col, name in zip([0, 1, 2, 3], CLUSTER_COLORS, CLUSTER_NAMES):
                mask = semantic_labels == sem_id
                if mask.sum() > 0:
                    ax.scatter(
                        df_plot['Sleep'].values[mask],
                        df_plot['Productivity'].values[mask],
                        color=col, alpha=0.30, s=16, edgecolors='none',
                        label=name, zorder=2,
                    )
                    plotted_clusters.add(sem_id)

        except Exception:
            # Fallback: simple scatter without clusters
            ax.scatter(raw_df['Sleep'].values[:1000], raw_df['Productivity'].values[:1000],
                       color=CYAN, alpha=0.25, s=12, edgecolors='none', label='Population')

    # User point — bright star
    user_label = cluster_data['cluster_name'] if cluster_data else 'You'
    ax.scatter([user_sleep], [user_prod], color='#FFFF00', s=200, marker='*',
               zorder=8, edgecolors='white', linewidths=0.8, label=f'You ({user_label})')
    ax.annotate(
        f" ← You",
        xy=(user_sleep, user_prod), xytext=(user_sleep + 0.15, user_prod + 1.5),
        color='#FFFF00', fontsize=8.5, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#FFFF00', lw=1.2),
    )

    ax.set_xlabel('Sleep (hours)', color=TEXT_COLOR, fontsize=9.5)
    ax.set_ylabel('Productivity Score', color=TEXT_COLOR, fontsize=9.5)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.grid(True, alpha=0.12, linestyle='--')

    legend = ax.legend(
        facecolor=CARD_BG, edgecolor=MUTED_TEXT, labelcolor=TEXT_COLOR,
        fontsize=8, loc='lower right', framealpha=0.9,
    )

    ax.set_title('Lifestyle Cluster Map', color=TEXT_COLOR, fontsize=12, fontweight='bold', pad=10)
    return _fig_to_base64(fig)


def generate_scatter_plots(df, user_data):
    """7. 2x2 Scatter Plots with Linear Regression lines."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), facecolor=BG_COLOR)
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    pairs = [
        ('Sleep', 'Productivity', float(user_data.get('sleep', 7.5)), CYAN),
        ('Stress', 'Productivity', float(user_data.get('stress', 4)), RED),
        ('StudyHours', 'Productivity', float(user_data.get('study', 3.5)), PURPLE),
        ('Water', 'Productivity', float(user_data.get('water', 2.0)), GREEN)
    ]

    for ax, (x_col, y_col, user_val, col) in zip(axes.flatten(), pairs):
        ax.set_facecolor(CARD_BG)
        if x_col in df.columns and y_col in df.columns:
            x = df[x_col].dropna().values[:500]
            y = df[y_col].dropna().values[:500]

            ax.scatter(x, y, alpha=0.25, color=col, s=15, edgecolors='none')

            # Regression line
            slope, intercept, _, _, _ = stats.linregress(x, y)
            line_x = np.linspace(x.min(), x.max(), 100)
            ax.plot(line_x, slope * line_x + intercept, color=TEXT_COLOR, linewidth=1.5, linestyle='--')

            # User current point star marker
            user_y = float(user_data.get('prediction', 85.0)) if user_data.get('prediction') else 85.0
            ax.scatter([user_val], [user_y], color='#FFFF00', s=120, marker='*', zorder=5, label='You')

        ax.set_xlabel(x_col, color=TEXT_COLOR, fontsize=8)
        ax.set_ylabel(y_col, color=TEXT_COLOR, fontsize=8)
        ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        ax.grid(True, alpha=0.15)

    fig.suptitle('Habit Correlation Scatter & Regression Analysis', color=TEXT_COLOR, fontsize=12, fontweight='bold')
    return _fig_to_base64(fig)


def generate_sankey_diagram(user_data, prediction):
    """8. Sankey Flow Diagram from Habits -> Lifestyle -> Productivity -> Life Score."""
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)

    sankey = Sankey(ax=ax, scale=0.01, offset=0.2, head_angle=120, format='%.0f', shoulder=0.05)

    # First flow node: Habit Inputs
    sleep_flow = float(user_data.get('sleep', 7.5)) * 5
    water_flow = float(user_data.get('water', 2.0)) * 10
    stress_sub = float(user_data.get('stress', 4)) * 4

    sankey.add(flows=[sleep_flow, water_flow, -stress_sub, - (sleep_flow + water_flow - stress_sub)],
               labels=['Sleep Energy', 'Hydration', 'Stress Drain', 'Lifestyle Output'],
               orientations=[1, 1, -1, 0],
               pathlengths=[0.2, 0.2, 0.2, 0.4],
               facecolor=CYAN,
               edgecolor=PURPLE)

    diagrams = sankey.finish()
    for text in diagrams[0].texts:
        text.set_color(TEXT_COLOR)
        text.set_fontsize(8)

    ax.axis('off')
    ax.set_title('Energy & Habit Flow Diagram', color=TEXT_COLOR, fontsize=12, fontweight='bold', pad=10)
    return _fig_to_base64(fig)


def generate_correlation_matrix(df):
    """9. Heatmap of all numerical features."""
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)

    num_cols = df.select_dtypes(include=[np.number]).columns[:9]
    corr = df[num_cols].corr()

    cax = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    cbar = fig.colorbar(cax, ax=ax)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)

    ax.set_xticks(np.arange(len(num_cols)))
    ax.set_yticks(np.arange(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha='right', color=TEXT_COLOR, fontsize=8)
    ax.set_yticklabels(num_cols, color=TEXT_COLOR, fontsize=8)

    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center',
                    color='#FFFFFF' if abs(corr.iloc[i, j]) > 0.4 else '#000000', fontsize=7)

    ax.set_title('Feature Correlation Matrix', color=TEXT_COLOR, fontsize=12, fontweight='bold', pad=10)
    return _fig_to_base64(fig)


def generate_distribution_graphs(df, user_data):
    """10. Histograms with KDE for Sleep, Stress, Water, Study, Productivity."""
    fig, axes = plt.subplots(1, 5, figsize=(10, 2.5), facecolor=BG_COLOR)
    fig.subplots_adjust(wspace=0.35)

    cols = [
        ('Sleep', float(user_data.get('sleep', 7.5)), PURPLE),
        ('Stress', float(user_data.get('stress', 4)), RED),
        ('Water', float(user_data.get('water', 2.0)), GREEN),
        ('StudyHours', float(user_data.get('study', 3.5)), CYAN),
        ('Productivity', float(user_data.get('prediction', 85.0)) if user_data.get('prediction') else 85.0, PINK)
    ]

    for ax, (col_name, user_val, color) in zip(axes, cols):
        ax.set_facecolor(CARD_BG)
        if col_name in df.columns:
            data = df[col_name].dropna().values[:1000]
            ax.hist(data, bins=15, color=color, alpha=0.4, density=True, edgecolor='none')

            # Draw vertical user line
            ax.axvline(user_val, color='#FFFF00', linestyle='--', linewidth=1.8, label='You')

        ax.set_title(col_name, color=TEXT_COLOR, fontsize=9, fontweight='bold')
        ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        ax.grid(True, alpha=0.1)

    fig.suptitle('Population Distribution vs Your Input', color=TEXT_COLOR, fontsize=11, fontweight='bold', y=1.05)
    return _fig_to_base64(fig)


def generate_trend_analysis(user_df, user_pred=85.0):
    """11. Weekly & Monthly Trend Line Chart."""
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)

    if user_df is not None and not user_df.empty and 'productivity' in user_df.columns:
        weekly_scores = pd.to_numeric(user_df['productivity'].tail(30), errors='coerce').fillna(0).values
        dates = user_df['date'].tail(30).values
        days = [str(d).split('-')[-1] for d in dates]
    else:
        np.random.seed(99)
        days = [f"W{i+1}" for i in range(12)]
        weekly_scores = np.random.normal(user_pred, 3.5, 12)
        weekly_scores[-1] = user_pred

    moving_avg = pd.Series(weekly_scores).rolling(window=3, min_periods=1).mean()

    ax.plot(days, weekly_scores, marker='s', color=PINK, linewidth=2, label='Daily Life Score')
    ax.plot(days, moving_avg, linestyle='-', color=CYAN, linewidth=2.5, label='3-Day Moving Avg')

    ax.fill_between(days, weekly_scores - 2, weekly_scores + 2, color=PINK, alpha=0.15)

    ax.set_ylabel('Life Score Rating', color=TEXT_COLOR, fontsize=9)
    ax.set_xticklabels(days, rotation=45, color=TEXT_COLOR, fontsize=8)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, alpha=0.15)
    ax.legend(facecolor=CARD_BG, edgecolor=MUTED_TEXT, labelcolor=TEXT_COLOR, fontsize=8)

    ax.set_title('Long-term Life Score Trend Analysis', color=TEXT_COLOR, fontsize=12, fontweight='bold', pad=10)
    return _fig_to_base64(fig)


def generate_feature_importance_chart(model, feature_columns):
    """12. Feature Importance Horizontal Bar Plot."""
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.ones(len(feature_columns)) / len(feature_columns)

    indices = np.argsort(importances)
    sorted_cols = [feature_columns[i] for i in indices]
    sorted_imps = [importances[i] * 100 for i in indices]

    colors = [CYAN if imp > 15 else PURPLE for imp in sorted_imps]

    bars = ax.barh(np.arange(len(sorted_cols)), sorted_imps, color=colors, height=0.6)

    ax.set_yticks(np.arange(len(sorted_cols)))
    ax.set_yticklabels(sorted_cols, color=TEXT_COLOR, fontsize=9)
    ax.set_xlabel('Model Relative Contribution (%)', color=TEXT_COLOR, fontsize=9)
    ax.tick_params(colors=TEXT_COLOR)

    for bar, val in zip(bars, sorted_imps):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2.0, f"{val:.1f}%",
                va='center', color=TEXT_COLOR, fontsize=8, fontweight='bold')

    ax.set_title('RandomForest Feature Importance Analysis', color=TEXT_COLOR, fontsize=12, fontweight='bold', pad=10)
    return _fig_to_base64(fig)


def generate_mini_line(values, color=CYAN):
    fig, ax = plt.subplots(figsize=(1.8, 0.45), facecolor='none')
    ax.set_facecolor('none')
    if len(values) < 2:
        values = [values[0], values[0]] if values else [0.0, 0.0]
    ax.plot(values, color=color, linewidth=2)
    ax.axis('off')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return _fig_to_base64(fig)


def generate_mini_bar(values, color=PURPLE):
    fig, ax = plt.subplots(figsize=(1.8, 0.45), facecolor='none')
    ax.set_facecolor('none')
    if not values:
        values = [0.0]
    ax.bar(np.arange(len(values)), values, color=color, width=0.6, edgecolor='none')
    ax.axis('off')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return _fig_to_base64(fig)


def generate_mini_progress(pct, color=GREEN):
    fig, ax = plt.subplots(figsize=(1.8, 0.45), facecolor='none')
    ax.set_facecolor('none')
    ax.barh(0, 100, color='#2A1748', height=0.3, edgecolor='none')
    ax.barh(0, pct, color=color, height=0.3, edgecolor='none')
    ax.axis('off')
    ax.set_xlim(-5, 105)
    ax.set_ylim(-0.5, 0.5)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return _fig_to_base64(fig)


def generate_mini_donut(pct, color=CYAN):
    fig, ax = plt.subplots(figsize=(0.9, 0.45), facecolor='none')
    ax.set_facecolor('none')
    ax.pie([pct, max(0, 100 - pct)], colors=[color, '#2A1748'], startangle=90, 
           wedgeprops=dict(width=0.4, edgecolor='none'))
    ax.axis('off')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return _fig_to_base64(fig)


def generate_mood_calendar_chart(user_df):
    """Mood breakdown distribution (represented as categories for calendar feel)."""
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)
    if user_df is not None and not user_df.empty and 'mood' in user_df.columns:
        moods = pd.to_numeric(user_df['mood'].tail(30), errors='coerce').fillna(0).values
    else:
        moods = [7] * 10
    
    categories = ['Very Bad', 'Okay', 'Neutral', 'Good', 'Excellent']
    counts = [0, 0, 0, 0, 0]
    for m in moods:
        if m <= 2: counts[0] += 1
        elif m <= 4: counts[1] += 1
        elif m <= 6: counts[2] += 1
        elif m <= 8: counts[3] += 1
        else: counts[4] += 1
        
    ax.bar(categories, counts, color=PINK, width=0.5)
    ax.tick_params(colors=TEXT_COLOR)
    ax.set_ylabel('Day Count', color=TEXT_COLOR, fontsize=9)
    ax.grid(True, alpha=0.1)
    ax.set_title('Mood Frequency Distribution (Past 30 Days)', color=TEXT_COLOR, fontsize=12, fontweight='bold', pad=10)
    return _fig_to_base64(fig)


def generate_mood_timeline_chart(user_df):
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)
    if user_df is not None and not user_df.empty and 'mood' in user_df.columns:
        moods = pd.to_numeric(user_df['mood'].tail(14), errors='coerce').fillna(0).values
        dates = user_df['date'].tail(14).values
        days = [str(d).split('-')[-1] for d in dates]
    else:
        days = [f"D{i+1}" for i in range(14)]
        moods = [7] * 14
        
    ax.plot(days, moods, marker='o', color=PINK, linewidth=2, label='Mood Level')
    ax.axhline(7.0, color=CYAN, linestyle=':', label='Target Mood')
    ax.set_ylabel('Mood Level', color=TEXT_COLOR, fontsize=9)
    ax.set_xticklabels(days, rotation=45, color=TEXT_COLOR, fontsize=8)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, alpha=0.1)
    ax.legend(facecolor=CARD_BG, edgecolor=MUTED_TEXT, labelcolor=TEXT_COLOR, fontsize=8)
    ax.set_title('Daily Mood Timeline (Past 14 Days)', color=TEXT_COLOR, fontsize=12, fontweight='bold', pad=10)
    return _fig_to_base64(fig)


def generate_expense_donut(user_df, user_expense):
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)
    
    today_exp = float(user_expense)
    if today_exp == 0:
        today_exp = 50.0
    food = today_exp * 0.3
    travel = today_exp * 0.25
    bills = today_exp * 0.35
    leisure = today_exp * 0.1
    
    sizes = [food, travel, bills, leisure]
    labels = ['Food', 'Travel', 'Bills', 'Leisure']
    colors = [CYAN, PURPLE, PINK, ORANGE]
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140,
                                      wedgeprops=dict(width=0.4, edgecolor='none'))
                                      
    for t in texts:
        t.set_color(TEXT_COLOR)
        t.set_fontsize(8)
    for at in autotexts:
        at.set_color('#000000')
        at.set_fontsize(8)
        at.set_weight('bold')
        
    ax.set_title("Today's Expense Breakdown", color=TEXT_COLOR, fontsize=12, fontweight='bold', pad=10)
    return _fig_to_base64(fig)


def generate_monthly_spending(user_df):
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)
    if user_df is not None and not user_df.empty and 'expense' in user_df.columns:
        expenses = pd.to_numeric(user_df['expense'].tail(30), errors='coerce').fillna(0).values
        dates = user_df['date'].tail(30).values
        days = [str(d).split('-')[-1] for d in dates]
    else:
        days = [f"D{i+1}" for i in range(15)]
        expenses = [45] * 15
        
    ax.plot(days, expenses, marker='o', color=ORANGE, linewidth=2, label='Daily Spending ($)')
    ax.fill_between(days, expenses, color=ORANGE, alpha=0.1)
    ax.set_ylabel('Spending ($)', color=TEXT_COLOR, fontsize=9)
    ax.set_xticklabels(days, rotation=45, color=TEXT_COLOR, fontsize=8)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, alpha=0.1)
    ax.set_title('Spending Trend (Past 30 Days)', color=TEXT_COLOR, fontsize=12, fontweight='bold', pad=10)
    return _fig_to_base64(fig)


def generate_screentime_bars(user_df):
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)
    if user_df is not None and not user_df.empty and 'screentime' in user_df.columns:
        screentimes = pd.to_numeric(user_df['screentime'].tail(7), errors='coerce').fillna(0).values
        dates = user_df['date'].tail(7).values
        days = [str(d).split('-')[-1] for d in dates]
    else:
        days = [f"D{i+1}" for i in range(7)]
        screentimes = [4.2] * 7
        
    ax.bar(days, screentimes, color=PURPLE, width=0.4, label='Daily Screen Time')
    ax.axhline(4.0, color=CYAN, linestyle='--', label='Focus Limit (4h)')
    ax.set_ylabel('Hours', color=TEXT_COLOR, fontsize=9)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, alpha=0.1)
    ax.legend(facecolor=CARD_BG, edgecolor=MUTED_TEXT, labelcolor=TEXT_COLOR, fontsize=8)
    ax.set_title('Screen Time (Last 7 Days)', color=TEXT_COLOR, fontsize=12, fontweight='bold', pad=10)
    return _fig_to_base64(fig)


def generate_whatif_chart(whatif_data):
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)
    
    categories = ['Current', '+1.5h Sleep', '-2h Screen', 'All Optimized']
    base = 0.0
    impacts = [base, whatif_data.get('opt_sleep_diff', 0), whatif_data.get('opt_screen_diff', 0), whatif_data.get('opt_both_diff', 0)]
    
    colors = [PURPLE, CYAN, PINK, GREEN]
    bars = ax.bar(categories, impacts, color=colors, width=0.4, edgecolor='none')
    
    ax.axhline(0, color=TEXT_COLOR, linewidth=0.8, linestyle='-')
    ax.set_ylabel('Score Shift (points)', color=TEXT_COLOR, fontsize=9)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, alpha=0.1)
    
    for bar, val in zip(bars, impacts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, height + (0.3 if height >= 0 else -1.2), f"{val:+.1f}",
                ha='center', va='bottom', color=TEXT_COLOR, fontsize=9, fontweight='bold')
                
    ax.set_title('What-If Action Impact Projections', color=TEXT_COLOR, fontsize=12, fontweight='bold', pad=10)
    return _fig_to_base64(fig)
