import os
import json
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import pickle
from shapely.geometry import shape

# Suppress UserWarning from Matplotlib and other warnings
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')
warnings.filterwarnings("ignore", category=FutureWarning)


# --- Configuration ---
RESULTS_DIR = 'results_segregation_20250713'
TRAJECTORIES_DIR = os.path.join(RESULTS_DIR, 'trajectories')
MAP_FILE = os.path.join('cache', 'map_data_Chicago.pkl')
OUTPUT_DIR = 'visualization_results_by_step'
NUM_CBGS_TO_VISUALIZE = 10

# --- Create output directory ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 1. Load Map Data (using improved geometry parsing) ---
print(f"Loading map data from {MAP_FILE}...")
try:
    with open(MAP_FILE, 'rb') as f:
        chicago_data = pickle.load(f)

    cbg_geometries = []
    if 'cbgs' in chicago_data and isinstance(chicago_data['cbgs'], dict):
        for cbg_id, cbg_info in tqdm(chicago_data['cbgs'].items(), desc="Processing CBG shapes"):
            try:
                geom = None
                # Try different geometry fields and formats
                if 'shapely_lnglat' in cbg_info and cbg_info['shapely_lnglat'] is not None:
                    geom = cbg_info['shapely_lnglat']
                elif 'geometry' in cbg_info:
                    if isinstance(cbg_info['geometry'], pd.Series) and len(cbg_info['geometry']) > 0:
                        geom_obj = cbg_info['geometry'].iloc[0]
                        if hasattr(geom_obj, 'geoms'):
                            geom = list(geom_obj.geoms)[0]
                        else:
                            geom = geom_obj
                    elif isinstance(cbg_info['geometry'], dict):
                        geom = shape(cbg_info['geometry'])
                    elif isinstance(cbg_info['geometry'], str):
                        try:
                            geom_data = json.loads(cbg_info['geometry'])
                            geom = shape(geom_data)
                        except json.JSONDecodeError:
                            continue
                
                if geom is not None and not geom.is_empty:
                    cbg_geometries.append({
                        'cbg_id': str(cbg_id).strip(),
                        'geometry': geom
                    })
            except Exception as e:
                print(f"Error processing CBG {cbg_id}: {str(e)}")
                continue
    
    if not cbg_geometries:
        raise ValueError("Could not extract any valid CBG geometries from the map file.")

    map_gdf = gpd.GeoDataFrame(cbg_geometries, geometry='geometry', crs="EPSG:4326")
    print(f"Map data loaded successfully. Found {len(map_gdf)} CBG shapes.")

except FileNotFoundError:
    print(f"Error: Map file not found at {MAP_FILE}.")
    exit()
except Exception as e:
    print(f"An error occurred while loading or processing the map file: {e}")
    exit()

# --- 2. Load all agent data ---
print("Loading agent simulation data...")
agent_files = [f for f in os.listdir(TRAJECTORIES_DIR) if f.startswith('R') and f.endswith('.json')]
all_agents_data = []

for filename in tqdm(agent_files, desc="Processing agent files"):
    filepath = os.path.join(TRAJECTORIES_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            all_agents_data.append(data)
        except json.JSONDecodeError:
            continue

# --- 3. Process data to track agent locations and types over time ---
print("Processing agent trajectories...")
agent_locations_over_time = []
max_step = 0

for agent_data in all_agents_data:
    agent_id = agent_data['agent_profile']['agent_id']
    agent_type = agent_data['agent_profile']['agent_type']
    income_level = agent_data['agent_profile'].get('income_level', 'Unknown')
    race = agent_data['agent_profile'].get('race', 'Unknown')
    initial_residence = agent_data['initial_residence']
    
    # Add initial state
    agent_locations_over_time.append({
        'step': 0,
        'agent_id': agent_id,
        'agent_type': agent_type,
        'income_level': income_level,
        'race': race,
        'cbg_id': initial_residence,
        'utility': None
    })

    # Add states after migrations
    for migration in agent_data.get('migration_history', []):
        step = migration.get('step', 0)
        max_step = max(max_step, step + 1)
        agent_locations_over_time.append({
            'step': step + 1,
            'agent_id': agent_id,
            'agent_type': agent_type,
            'income_level': income_level,
            'race': race,
            'cbg_id': migration['to_cbg'],
            'utility': migration.get('utility_context', {}).get('agent_utility_after')
        })

df = pd.DataFrame(agent_locations_over_time)
df['is_altruist'] = (df['agent_type'] == 'altruist').astype(int)
df['is_egoist'] = (df['agent_type'] == 'egoist').astype(int)

# --- 4. Prepare data for plotting ---
print("Preparing data for plotting...")
cbgs_in_simulation = df['cbg_id'].unique()
cbgs_to_plot = sorted([cbg for cbg in cbgs_in_simulation if cbg in map_gdf['cbg_id'].values])[:NUM_CBGS_TO_VISUALIZE]
print(f"Will generate plots for {len(cbgs_to_plot)} CBGs found in both simulation and map data.")

system_utility_over_time = df.groupby('step')['utility'].mean().dropna()

cbg_metrics = df.groupby(['step', 'cbg_id']).agg(
    num_agents=('agent_id', 'count'),
    num_altruists=('is_altruist', 'sum'),
    num_egoists=('is_egoist', 'sum'),
    avg_utility=('utility', 'mean')
).reset_index()

all_steps = range(max_step + 1)
full_index = pd.MultiIndex.from_product([all_steps, cbgs_to_plot], names=['step', 'cbg_id'])
cbg_metrics = cbg_metrics.set_index(['step', 'cbg_id']).reindex(full_index, fill_value=0).reset_index()

cbg_metrics['altruist_prop'] = cbg_metrics.apply(
    lambda row: row['num_altruists'] / row['num_agents'] if row['num_agents'] > 0 else 0, axis=1)
cbg_metrics['egoist_prop'] = cbg_metrics.apply(
    lambda row: row['num_egoists'] / row['num_agents'] if row['num_agents'] > 0 else 0, axis=1)

plot_data = map_gdf[map_gdf['cbg_id'].isin(cbgs_to_plot)].merge(cbg_metrics, on='cbg_id')

# --- 5. Generate one plot per step ---
print("Generating plots for each simulation step...")
for step in tqdm(all_steps, desc="Generating step plots"):
    step_data = plot_data[plot_data['step'] == step]
    
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(25, 12))
    fig.subplots_adjust(hspace=0.4, wspace=0.1)
    axes = axes.flatten()

    current_system_utility = system_utility_over_time.get(step, 0.0)
    fig.suptitle(
        f'Simulation Step: {step}\nOverall System Utility: {current_system_utility:.4f}',
        fontsize=24, fontweight='bold'
    )

    for i, cbg_id in enumerate(cbgs_to_plot):
        ax = axes[i]
        cbg_plot_data = step_data[step_data['cbg_id'] == cbg_id]
        
        # Plot the base shape from map_gdf first
        base_shape = map_gdf[map_gdf['cbg_id'] == cbg_id]
        base_shape.plot(ax=ax, facecolor='lightgray', edgecolor='black', linewidth=0.5)

        if not cbg_plot_data.empty:
            cbg_plot_instance = cbg_plot_data.iloc[0]
            egoist_prop = cbg_plot_instance['egoist_prop']
            altruist_prop = cbg_plot_instance['altruist_prop']
            utility = cbg_plot_instance['avg_utility']
            
            # Overlay the colored shape based on egoist proportion
            base_shape.plot(ax=ax, facecolor='red', alpha=egoist_prop, edgecolor='black', linewidth=0.8)

            info_text = (f"Utility: {utility:.3f}\n"
                         f"Egoists: {egoist_prop:.1%} | Altruists: {altruist_prop:.1%}")
        else:
            # If no data for this step, it remains lightgray
            info_text = "No agents in this CBG"
            
        ax.set_title(f"CBG: {cbg_id}", fontsize=12, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(info_text, fontsize=10, labelpad=10)

    for i in range(len(cbgs_to_plot), len(axes)):
        axes[i].set_visible(False)

    plot_filename = os.path.join(OUTPUT_DIR, f'step_{step:03d}.png')
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)

# --- 6. Generate time series plots ---
def generate_time_series_plots(df, cbgs_to_plot, output_dir):
    print("Generating time series plots...")
    
    # Create subdirectory for time series plots
    ts_output_dir = os.path.join(output_dir, 'time_series')
    os.makedirs(ts_output_dir, exist_ok=True)
    
    # Calculate overall system utility once
    system_utility = df.groupby('step')['utility'].mean().dropna()
    
    # 1. Overall System Utility Plot
    plt.figure(figsize=(12, 6))
    plt.plot(system_utility.index, system_utility.values, marker='o', linewidth=2)
    plt.title('Overall System Utility Over Time')
    plt.xlabel('Simulation Step')
    plt.ylabel('Average Utility')
    plt.grid(True)
    plt.savefig(os.path.join(ts_output_dir, 'overall_utility.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Per CBG Metrics
    for cbg_id in tqdm(cbgs_to_plot, desc="Generating CBG-specific plots"):
        cbg_data = df[df['cbg_id'] == cbg_id]
        
        # Create a figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Agent Type Distribution (Area Plot)
        ax1 = axes[0, 0]
        cbg_agents = cbg_data.groupby('step').agg({
            'is_altruist': 'sum',
            'is_egoist': 'sum'
        }).fillna(0)
        
        # Calculate proportions
        total_agents = cbg_agents['is_altruist'] + cbg_agents['is_egoist']
        altruist_prop = cbg_agents['is_altruist'] / total_agents
        egoist_prop = cbg_agents['is_egoist'] / total_agents
        
        # Create stacked area plot
        ax1.fill_between(altruist_prop.index, 0, altruist_prop.values, 
                        label='Altruists', alpha=0.6)
        ax1.fill_between(altruist_prop.index, altruist_prop.values, 1, 
                        label='Egoists', alpha=0.6)
        ax1.set_title(f'Agent Type Distribution in CBG {cbg_id}')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Population Share')
        ax1.legend()
        ax1.grid(True)

        # CBG vs System Utility
        ax2 = axes[0, 1]
        cbg_utility = cbg_data.groupby('step')['utility'].mean()
        ax2.plot(system_utility.index, system_utility.values, label='System Utility', marker='o')
        ax2.plot(cbg_utility.index, cbg_utility.values, label=f'CBG {cbg_id} Utility', marker='o')
        ax2.set_title('CBG vs System Utility')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Average Utility')
        ax2.legend()
        ax2.grid(True)

        # Income Distribution (Area Plot)
        ax3 = axes[1, 0]
        income_data = cbg_data.groupby(['step', 'income_level'])['agent_id'].count().unstack(fill_value=0)
        if not income_data.empty:
            # Normalize to get proportions
            income_props = income_data.div(income_data.sum(axis=1), axis=0)
            # Plot stacked area
            income_props.plot(kind='area', stacked=True, ax=ax3, alpha=0.6)
            ax3.set_title(f'Income Distribution in CBG {cbg_id}')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Share of Population')
            ax3.grid(True)

        # Race Distribution (Area Plot)
        ax4 = axes[1, 1]
        race_data = cbg_data.groupby(['step', 'race'])['agent_id'].count().unstack(fill_value=0)
        if not race_data.empty:
            # Normalize to get proportions
            race_props = race_data.div(race_data.sum(axis=1), axis=0)
            # Plot stacked area
            race_props.plot(kind='area', stacked=True, ax=ax4, alpha=0.6)
            ax4.set_title(f'Race Distribution in CBG {cbg_id}')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Share of Population')
            ax4.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(ts_output_dir, f'cbg_{cbg_id}_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Time series plots have been saved to {ts_output_dir}")

# Call the time series plotting function after the main visualization
generate_time_series_plots(df, cbgs_to_plot, OUTPUT_DIR)

print("\nAll visualizations completed successfully!") 