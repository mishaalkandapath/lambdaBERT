from typing import *

from altair.utils.schemapi import Undefined
import numpy as np  # for array calculations
import pandas as pd  # for Dataframes
import altair as alt
from pandas import DataFrame
import matplotlib.pyplot as plt
import math
import string
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
from pprint import pprint

from tqdm import tqdm
rng = np.random.default_rng(12345)

from itertools import combinations, product
import networkx as nx

class CausalGraph:
    def __init__(self, edges, num_nodes=5, special_node="Y", active_nodes=[]):
        self.edges = edges # adjacency list representation
        self.reverse_edges = {node: [y for y in self.edges if node in self.edges[y]] for node in self.edges} #reverse adjacency matrix
        self.num_nodes = num_nodes
        self.active_nodes = active_nodes
        self.active_nodes_flat = []
        for item in self.active_nodes:
            if type(item) is not set:
                self.active_nodes_flat.append(item)
            else:
                for subitem in item:
                    self.active_nodes_flat.append(subitem)
        self.special_node = special_node

    def activate_node_non_temporal(self, node):
        raise DeprecationWarning("This method is deprecated. Use activate_node instead.")
        if node in self.active_nodes or \
            not set(self.reverse_edges[node]).issubset(self.active_nodes): # because these nodes cannot be interacted with on their own -- their conditional
            return # nothing to do here:
        self.active_nodes.add(node)
        #activate everything in its adjacency list entry that has not already been activated
        to_activate_queue = [n for n in self.edges[node] if n not in self.active_nodes]
        for node in to_activate_queue:
            self.activate_node(node)
        return self.active_nodes.copy()

    def activate_node(self, node):
        # nothing to do if the node is already activated
        if node in self.active_nodes_flat or \
            not set(self.reverse_edges[node]).issubset(self.active_nodes_flat): # or if the node is a dependent node
            return self.active_nodes.copy() # simply return the current environment

        self.active_nodes.append(node) # append the node
        self.active_nodes_flat.append(node) #maintain a flat list of activity membership

        to_activate_queue = [n for n in self.edges[node] if n not in self.active_nodes_flat] # all the nodes that can be activated

        while to_activate_queue: # temporal nesting
            to_act_list = []
            cur_add = []
            for n in to_activate_queue:
                to_act = self.activate_node_low_level(n)
                if to_act is not None:
                    to_act_list.extend(to_act)
                    cur_add.extend(n)
            if cur_add:
                self.active_nodes.append(frozenset(cur_add))
                self.active_nodes_flat.extend(cur_add)
            to_activate_queue = to_act_list

        return self.active_nodes.copy()

    def activate_node_low_level(self, node):
        if node in self.active_nodes_flat or \
            not set(self.reverse_edges[node]).issubset(self.active_nodes_flat):
            return

        to_activate_queue = [n for n in self.edges[node] if n not in self.active_nodes_flat]
        return to_activate_queue

    def outcome_possible(self, outcome_nodes):
        #return set(outcome_nodes) == set(self.active_nodes) # no order effect -- chain == conf
        return outcome_nodes == self.active_nodes

    def reset_graph(self):
        self.active_nodes = []
        self.active_nodes_flat = []

    def get_graph_copy(self):
        return CausalGraph(self.edges.copy(), self.num_nodes, self.special_node, self.active_nodes.copy())
    # Vizualization
    def _calculate_arrow_position(self, x_points: List[float], y_points: List[float],
                            padding: float = 0.15) -> Tuple[float, float, float]:
        """
        Calculate position and rotation for arrow markers on curve, with padding from endpoint.
        padding: how far from the end node to place the arrow (0-1)
        """
        # Get points near the end, but not exactly at the end
        arrow_idx = int(len(x_points) * (1 - padding))
        base_idx = arrow_idx - 1

        x2, y2 = x_points[arrow_idx], y_points[arrow_idx]
        x1, y1 = x_points[base_idx], y_points[base_idx]

        dx = x2 - x1
        dy = y2 - y1
        angle = np.arctan2(dy, dx) * 180 / np.pi

        return x2, y2, angle

    def _generate_curve_points(self, x1: float, y1: float, x2: float, y2: float,
                            direction: int = 1, num_points: int = 100) -> Tuple[List[float], List[float]]:
        """
        Generate points for a quadratic curve between two points.
        Increased num_points for smoother curves and better arrow placement.
        """
        # Calculate midpoint
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        # Calculate normal vector to the line
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)

        if length == 0:
            return [x1], [y1]

        # Normal vector
        nx = -dy / length
        ny = dx / length

        # Control point offset from midpoint
        offset = 0.3 * direction  # Increased curve intensity slightly
        ctrl_x = mid_x + offset * nx
        ctrl_y = mid_y + offset * ny

        # Generate points along quadratic Bezier curve
        t = np.linspace(0, 1, num_points)
        x = (1-t)**2 * x1 + 2*(1-t)*t * ctrl_x + t**2 * x2
        y = (1-t)**2 * y1 + 2*(1-t)*t * ctrl_y + t**2 * y2

        return x.tolist(), y.tolist()

    def see_graph(self) -> alt.Chart:
        # Create a NetworkX graph for layout calculation
        G = nx.DiGraph()
        for source, targets in self.edges.items():
            for target in targets:
                G.add_edge(source, target)

        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Prepare node data
        nodes_data = []
        for node in G.nodes():
            x, y = pos[node]
            status = 'active' if node in self.active_nodes else 'inactive'
            if node == self.special_node:
                status = 'special'
            nodes_data.append({
                'node': node,
                'x': x,
                'y': y,
                'status': status
            })

        # Prepare edge data with curved paths for antiparallel edges
        edges_data = []
        arrows_data = []

        # Keep track of processed edges to handle antiparallel edges
        processed_edges = set()

        for source, targets in self.edges.items():
            for target in targets:
                if (source, target) in processed_edges:
                    continue

                source_pos = pos[source]
                target_pos = pos[target]

                # Check if there's an antiparallel edge
                is_antiparallel = target in self.edges and source in self.edges[target]

                # Generate curve points
                if is_antiparallel:
                    # Generate curved paths for both directions
                    x1_points, y1_points = self._generate_curve_points(
                        source_pos[0], source_pos[1],
                        target_pos[0], target_pos[1],
                        direction=1
                    )
                    x2_points, y2_points = self._generate_curve_points(
                        target_pos[0], target_pos[1],
                        source_pos[0], source_pos[1],
                        direction=-1
                    )
                    processed_edges.add((target, source))
                else:
                    # Straight line with slight curve
                    x1_points, y1_points = self._generate_curve_points(
                        source_pos[0], source_pos[1],
                        target_pos[0], target_pos[1],
                        direction=0.1
                    )

                # Add first edge
                for i in range(len(x1_points)-1):
                    edges_data.append({
                        'x': x1_points[i],
                        'y': y1_points[i],
                        'order': i,
                        'edge_id': f"{source}->{target}"
                    })

                # Add arrow for first edge
                arrow_x, arrow_y, angle = self._calculate_arrow_position(x1_points, y1_points)
                arrows_data.append({
                    'source': source,
                    'target': target,
                    'x': arrow_x,
                    'y': arrow_y,
                    'angle': angle
                })

                # Add second edge if antiparallel
                if is_antiparallel:
                    for i in range(len(x2_points)-1):
                        edges_data.append({
                            'x': x2_points[i],
                            'y': y2_points[i],
                            'order': i,
                            'edge_id': f"{target}->{source}"
                        })

                    # Add arrow for second edge
                    arrow_x, arrow_y, angle = self._calculate_arrow_position(x2_points, y2_points)
                    arrows_data.append({
                        'source': target,
                        'target': source,
                        'x': arrow_x,
                        'y': arrow_y,
                        'angle': angle
                    })

        # Create DataFrames
        nodes_df = pd.DataFrame(nodes_data)
        edges_df = pd.DataFrame(edges_data)
        arrows_df = pd.DataFrame(arrows_data)

        edge_lines = alt.Chart(edges_df).mark_line(
            color='gray',
            strokeWidth=1,
            opacity=0.6  # Slightly transparent edges
        ).encode(
            x=alt.X('x:Q', scale=alt.Scale(domain=[-1.5, 1.5])),
            y=alt.Y('y:Q', scale=alt.Scale(domain=[-1.5, 1.5])),
            detail='edge_id:N',
            order='order:Q',
            tooltip=['edge_id:N']
        )

        # Create nodes layer
        nodes = alt.Chart(nodes_df).mark_circle(
            size=300,
            tooltip=True
        ).encode(
            x='x',
            y='y',
            color=alt.Color('status:N',
                        scale=alt.Scale(
                            domain=['inactive', 'active', 'special'],
                            range=['lightblue', 'red', 'green']
                        )),
            tooltip=['node', 'status']
        )

        # Add node labels
        text = alt.Chart(nodes_df).mark_text(
            baseline='middle',
            align='center',
            fontSize=12,
            fontWeight='bold',
            color='white'
        ).encode(
            x='x',
            y='y',
            text='node'
        )

        # Create arrow markers - now with larger size and higher zIndex
        arrows = alt.Chart(arrows_df).mark_point(
            shape='triangle',
            filled=True,
            size=200,  # Increased size
            color='black',  # Changed to black for better visibility
            opacity=0.8
        ).encode(
            x='x:Q',
            y='y:Q',
            angle='angle:Q',
            tooltip=['source', 'target']
        )

        # Combine all layers with arrows on top
        chart = alt.layer(
            edge_lines,  # Bottom layer
            nodes,      # Middle layer
            text,       # Text layer
            arrows      # Top layer
        ).properties(
            width=400,
            height=300,
            title='Causal Graph Visualization'
        ).configure_view(
            strokeWidth=0
        ).configure_axis(
            grid=False,
            labels=False,
            ticks=False,
            domain=False
        )

        return chart

def make_graph(causal_graph, x1, x2):
    edges = {"A": [], "B": [], "C": [], "D": [], "Y": []}
    if (x1 not in ["A", "B", "C", "D"]) or (x2 not in ["A", "B", "C", "D"]):
        raise ValueError(f"Invalid inputs: x1 ({x1}) and x2 ({x2}) must be in [A, B, C, D]")
    match causal_graph:
        case "Parallel":
            edges[x1].append("Y")
            edges[x2].append("Y")
        case "Chain":
            edges[x1].append(x2)
            edges[x2].append("Y")
        case "Confound":
            edges[x1].extend([x2, "Y"])
        case _:
            raise ValueError(f"Invalid causal_graph {causal_graph}")
    return CausalGraph(edges, active_nodes=[])

def generate_all_graphs(vertices=["A", "B", "C", "D", "Y"]):
    graphs = []

    def create_edge_combinations():
        vertex_pairs = []
        for v1, v2 in combinations(vertices, 2):
            if vertices[-1] in [v1, v2]:
                if v1 == vertices[-1]: # special edge -- lightbulb is communicated to not activate anything?
                    vertex_pairs.append((v2, v1))
                else:
                    vertex_pairs.append((v1, v2))
            else:
                vertex_pairs.append((v1, v2))
                vertex_pairs.append((v2, v1))
        return product([True, False], repeat=len(vertex_pairs)), vertex_pairs

    def create_edge_dict(combination, pairs):
        # Initialize empty edge lists for each vertex
        edges = {vertex: [] for vertex in vertices}

        # Add edges based on the True/False combination
        for (v1, v2), include_edge in zip(pairs, combination):
            if include_edge:
                edges[v1].append(v2)

        return edges

    # Generate all possible graphs
    graphs = []
    edge_combinations, vertex_pairs = create_edge_combinations()

    for combination in edge_combinations:
        edges = create_edge_dict(combination, vertex_pairs)
        graphs.append(CausalGraph(edges))

    return graphs

def detect_cycle(edge_adj_list):
    visited = set()
    rec_stack = set()
    def dfs(node):
        if node in rec_stack:
            return True
        if node in visited:
            return False
        visited.add(node)
        rec_stack.add(node)
        for n in edge_adj_list[node]:
            if dfs(n):
                return True
        rec_stack.remove(node)
        return False
    for node in edge_adj_list:
        if dfs(node):
            return True
    return False

def repetitive_graph(nodes):
  # current "source"
  # first init source
  cur_sources = {k: [] for k in nodes.keys()}

  for source in nodes.keys():
    for receive in nodes[source]: cur_sources[receive].append(source)
  orig_finished = {node for node in cur_sources.keys() if cur_sources[node] == []}
  num_finished = len(orig_finished)

  while num_finished < len(nodes.keys()):
    num_finished = 0
    for node in nodes.keys():
      unoriginal_sources = set(cur_sources[node]) - orig_finished
      if len(cur_sources[node]) != len(set(cur_sources[node])): return None
      elif unoriginal_sources != set():
        for source in unoriginal_sources:
          assert cur_sources[source] != [], f"Error: An original source missed, graph: {nodes}"
          cur_sources[node].remove(source)
          if source in cur_sources[node]: return None
          cur_sources[node].extend(cur_sources[source]) # if cur_source has redundant sources, this will be checked at some point in the future

      else: num_finished += 1

  return nodes

def init_graphs():
  all_graphs = generate_all_graphs()
  rem_graphs = []
  cycle_graphs = []
  bad_graphs = []
  removals = [{"A":["B", "Y"], "B": ["Y"], "C": [], "D": [], "Y": []},]
  for g in tqdm(all_graphs):
      if g.edges in removals:
          bad_graphs.append(g)
          continue
      if detect_cycle(g.edges):
          cycle_graphs.append(g)
          continue

  for g in cycle_graphs:
      all_graphs.remove(g)
  for g in bad_graphs:
      all_graphs.remove(g)

  for g in tqdm(all_graphs):
      if repetitive_graph(g.edges) is None:
          rem_graphs.append(g)

  for g in rem_graphs:
      all_graphs.remove(g)

  print("Number of remaining graphs: ", len(all_graphs))
  return all_graphs

def max_entropy_reduction(real_graph, priors, step):
    max_reduction = float("-inf")
    b_action = None
    s_ig = None
    s_node = step["node"]

    for node in ["A", "B", "C", "D", "Y"]:
        rg_copy = real_graph.get_graph_copy()
        p_copy = {g.get_graph_copy(): p for g, p in priors.items()}
        g_copy = list(p_copy.keys())

        ig, _ = one_step_intervention_entropy(rg_copy, g_copy, p_copy, {"node": node})

        if ig > max_reduction:
            max_reduction = ig
            b_action = node
        if s_node == node:
            s_ig = ig
    # print("b_action", b_action, "but took ", s_node)
    s_outcome = real_graph.activate_node(step["node"])
    [g.activate_node(step["node"]) for g in priors]

    return max_reduction, s_ig, s_outcome


def one_step_intervention_entropy(real_graph, graphs, priors, step):
    """
    Calculate the quality of the intervention for each graph at the given step.
    """
    to_activate = step["node"]
    prev_ent = -sum(p*math.log(p, 2) for p in priors.values() if p!= 0)

    outcome = real_graph.activate_node(to_activate)
    outcomes = set([tuple(g.activate_node(to_activate)) for g in graphs])

    expected_ig = 0
    for out in outcomes:
        temp_priors = {g: p for g, p in priors.items()}
        for graph in [g for g in graphs if temp_priors[g] != 0]:
            if not graph.outcome_possible(list(out)):
                temp_priors[graph] = 0
        cur_entropy = -sum(p*math.log(p, 2) for p in temp_priors.values() if p!= 0)
        expected_ig += (prev_ent - cur_entropy) * sum(p for p in temp_priors.values() if p != 0)

    return expected_ig, outcome


def intervention_entropy_mean(real_graph, graphs, priors, steps, forgetting_rate=0.2, printing=False):
    """
    Calculate the mean entropy of the intervention for each graph.
    """
    efficiencies = []
    ig_acc, max_ig_acc = 0, 0
    # for step in tqdm(steps.iterrows(), total=len(steps)):
    for step in steps.iterrows():
        max_ig, ig, t_outcome = max_entropy_reduction(real_graph, priors, step[1])
        ig_acc += ig
        max_ig_acc += max_ig

        if printing: print("Cur exp ig {} max ig {}".format(ig, max_ig))
        for graph in graphs:
            if not graph.outcome_possible(t_outcome):
                priors[graph] = 0
        assert any(g.edges == real_graph.edges for g in graphs), "Real graph isnt there !!!"
        assert all(p for g, p in priors.items() if g.edges == real_graph.edges), "Not real graph!!!"
        assert any(p != 0 for p in priors.values()), "No possible outcomes left"

        if "Y" in real_graph.active_nodes_flat:
            if max_ig_acc != 0: efficiencies.append(ig_acc/max_ig_acc)
            else: efficiencies.append(0.0)
            ig_acc, max_ig_acc = 0, 0
            if printing: print("Round Efficiency: ", efficiencies[-1])
            real_graph.reset_graph()
            [g.reset_graph() for g in graphs]
            priors = {g: (1-forgetting_rate)*p + forgetting_rate/len(graphs) for g, p in priors.items()}

        if len([p for p in priors.values() if p != 0]) == 1:
            break

    if max_ig_acc != 0:
        efficiencies.append(ig_acc/max_ig_acc)
        if printing: print("Round Efficiency: ", efficiencies[-1])
    return np.mean(efficiencies)

def calc_intervention_per_participant(participant_data, forgetting_rate=0, printing=False):
    #remove all rows where node value is Start or scenario is Familiarization
    intervention_df = []
    participant_data = participant_data[participant_data["node"] != "Start"]
    participant_data = participant_data[participant_data["scenario"] != "Familiarization"]

    # get the different groups, which are trials, by grouping on scenario and graph columns
    groups = participant_data.groupby(["scenario", "graph"])
    # run per trial
    graphs = all_graphs
    for name, group in (groups):
        scenario, cg = name
        sample_row = participant_data.loc[(participant_data["scenario"] == scenario) & (participant_data["graph"] == cg)].iloc[0]
        if printing: print("Current details: ", scenario, cg, f"X1: {sample_row['endNode1']}, X2: {sample_row['endNode2']}")
        real_graph = make_graph(cg, sample_row['endNode1'], sample_row['endNode2'])
        if printing: print(real_graph.edges)
        [g.reset_graph() for g in graphs]
        priors = {g: 1/len(graphs) for g in graphs}
        iq_mean = intervention_entropy_mean(real_graph, graphs, priors, group, forgetting_rate)
        #append a row to df
        intervention_df.append({"scenario": scenario, "graph": cg, "intervention_quality": iq_mean})

    return pd.DataFrame(intervention_df)

import random
def generate_random_trials(df):
    nodes_to_tap = ["A", "B", "C", "D", "Y"]
    
    for index, row in df.iterrows():
        df.at[index, "node"] = random.choice(nodes_to_tap)
    
    return df
  

def random_participant(outdir, n=1000):
    """
    Generate random trials n times and average the intervention quality scores
    across iterations for each scenario and graph combination.
    
    Parameters:
    outdir (str): Output directory path
    n (int): Number of iterations to run
    
    Returns:
    pd.DataFrame: Averaged intervention quality scores
    """
    participant_df = pd.read_csv(f"examine.csv")
    
    # Initialize dictionary to store accumulated scores
    accumulated_scores = {}
    
    # Run iterations
    for i in tqdm(range(n)):
        random_df = generate_random_trials(participant_df)
        intervention_df = calc_intervention_per_participant(random_df)
        
        # Accumulate scores for each scenario-graph combination
        for _, row in intervention_df.iterrows():
            key = (row['scenario'], row['graph'])
            if key not in accumulated_scores:
                accumulated_scores[key] = {
                    'total_score': row['intervention_quality'],
                    'count': 1
                }
            else:
                accumulated_scores[key]['total_score'] += row['intervention_quality']
                accumulated_scores[key]['count'] += 1
    
    # Calculate averages and create final DataFrame
    final_data = []
    for (scenario, graph), scores in accumulated_scores.items():
        avg_score = scores['total_score'] / scores['count']
        final_data.append({
            'id': 999,
            'scenario': scenario,
            'graph': graph,
            'intervention_quality': avg_score
        })
    
    # Convert to DataFrame and sort
    final_df = pd.DataFrame(final_data)
    final_df = final_df.sort_values(['scenario', 'graph']).reset_index(drop=True)
    
    # Save to CSV
    final_df.to_csv(f"data_{999}_iq.csv", index=False)
    
    return final_df

all_graphs = init_graphs()

random_participant("")