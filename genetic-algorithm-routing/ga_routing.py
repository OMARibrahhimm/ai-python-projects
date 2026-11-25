import re
import random
import tkinter as tk
from tkinter import ttk, messagebox

import networkx as nx
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

MAX_GENERATIONS = 1000   # safety cap when the stopping criterion is "Saturation"



def parse_graph(text):
    """
    Adjacency list  (one node per line, weight in brackets, default weight = 1):
        A: B(4), C(2)
        B: D(5), F(10)
        F:
    Adjacency matrix (first line = node names, then one row per node, 0 = no edge):
        A B C
        0 4 2
        0 0 3
        0 0 0
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        raise ValueError("Graph description is empty.")

    G = nx.DiGraph()

    if ":" in lines[0]:                                   #adjacency list
        for line in lines:
            if ":" not in line:
                raise ValueError(f"Missing ':' in line: {line}")
            node, rest = line.split(":", 1)
            node = node.strip()
            if not re.fullmatch(r"\w+", node):
                raise ValueError(f"Bad node name: '{node}'")
            G.add_node(node)
            for token in rest.split(","):
                token = token.strip()
                if not token:
                    continue
                m = re.fullmatch(r"(\w+)\s*(?:\(\s*(\d+(?:\.\d+)?)\s*\))?", token)
                if not m:
                    raise ValueError(f"Cannot read '{token}' in line: {line}")
                neighbour = m.group(1)
                weight = float(m.group(2)) if m.group(2) else 1.0
                G.add_edge(node, neighbour, weight=weight)
    else:                                                 #adjacency matrix
        names = lines[0].split()
        rows = lines[1:]
        if len(rows) != len(names):
            raise ValueError("Matrix must have one row per node name.")
        G.add_nodes_from(names)
        for i, row in enumerate(rows):
            values = row.split()
            if len(values) != len(names):
                raise ValueError(f"Row {i + 1} must have {len(names)} values.")
            for j, v in enumerate(values):
                try:
                    w = float(v)
                except ValueError:
                    raise ValueError(f"'{v}' in row {i + 1} is not a number.")
                if w > 0 and i != j:
                    G.add_edge(names[i], names[j], weight=w)
    return G



def random_path(G, start, end, blocked=()):
    """Random walk from start to end without revisiting nodes. None if stuck."""
    path = [start]
    visited = set(blocked) | {start}
    while path[-1] != end:
        options = [n for n in G.successors(path[-1]) if n not in visited]
        if not options:
            return None
        nxt = random.choice(options)
        path.append(nxt)
        visited.add(nxt)
    return path


def remove_loops(path):
    """If a node appears twice, cut the loop between the two appearances."""
    out = []
    for n in path:
        if n in out:
            out = out[:out.index(n) + 1]
        else:
            out.append(n)
    return out


class PathProblem:
    """Chromosome = a valid path from start to end, e.g. ['A', 'C', 'F']."""

    def __init__(self, G, start, end):
        if not nx.has_path(G, start, end):
            raise ValueError(f"There is no path from {start} to {end} in this graph.")
        self.G, self.start, self.end = G, start, end

    def random_individual(self):
        for _ in range(5000):
            p = random_path(self.G, self.start, self.end)
            if p:
                return p
        return nx.shortest_path(self.G, self.start, self.end)   # fallback, always valid

    def route(self, ind):
        return ind

    def cost(self, ind):
        return sum(self.G[a][b]["weight"] for a, b in zip(ind, ind[1:]))

    def is_valid(self, ind):
        return True

    def crossover(self, p1, p2):
        """Find a common node, take p1 up to it and p2 after it."""
        common = [n for n in p1[1:-1] if n in p2[1:-1]]
        if not common:
            return p1[:]
        c = random.choice(common)
        return remove_loops(p1[:p1.index(c)] + p2[p2.index(c):])

    def mutate(self, ind):
        """Keep a prefix of the path and re-route the rest randomly."""
        if len(ind) < 2:
            return ind
        cut = random.randint(0, len(ind) - 2)
        prefix = ind[:cut + 1]
        tail = random_path(self.G, prefix[-1], self.end, blocked=prefix[:-1])
        return prefix[:-1] + tail if tail else ind


class TourProblem:

    def __init__(self, G, start, back_to_start):
        if G.number_of_nodes() < 2:
            raise ValueError("The graph needs at least 2 nodes.")
        self.G, self.start, self.back = G, start, back_to_start
        self.others = [n for n in G.nodes if n != start]
        # any valid tour uses each edge at most once, so it costs <= total weight
        self.penalty = sum(d["weight"] for _, _, d in G.edges(data=True)) + 1

    def random_individual(self):
        ind = self.others[:]
        random.shuffle(ind)
        return ind

    def route(self, ind):
        r = [self.start] + ind
        return r + [self.start] if self.back else r

    def cost(self, ind):
        r = self.route(ind)
        return sum(self.G[a][b]["weight"] if self.G.has_edge(a, b) else self.penalty
                   for a, b in zip(r, r[1:]))

    def is_valid(self, ind):
        r = self.route(ind)
        return all(self.G.has_edge(a, b) for a, b in zip(r, r[1:]))

    def crossover(self, p1, p2):
        """Order crossover: copy a slice of p1, fill the rest in p2's order."""
        n = len(p1)
        if n < 2:
            return p1[:]
        a, b = sorted(random.sample(range(n), 2))
        middle = p1[a:b + 1]
        rest = [g for g in p2 if g not in middle]
        return rest[:a] + middle + rest[a:]

    def mutate(self, ind):
        """Swap two random nodes."""
        ind = ind[:]
        if len(ind) >= 2:
            i, j = random.sample(range(len(ind)), 2)
            ind[i], ind[j] = ind[j], ind[i]
        return ind


def make_selector(population, fits, mode):
    """Returns a function that picks one parent each time it is called."""
    idx = range(len(population))
    if mode == "Ranking":
        order = sorted(idx, key=lambda i: fits[i])          # worst -> best
        weights = [0] * len(population)
        for rank, i in enumerate(order, start=1):           # worst gets 1, best gets N
            weights[i] = rank
        return lambda: population[random.choices(idx, weights=weights)[0]]
    # Tournament: pick k random individuals, the best one wins
    k = min(3, len(population))
    return lambda: population[max(random.sample(idx, k), key=lambda i: fits[i])]



           #GA LOOP

def run_ga(problem, pop_size, cx_rate, mut_rate, elit_rate,
           selection, stop_mode, stop_value):
    """
    cx_rate / mut_rate / elit_rate are fractions (0..1).
    stop_mode: "generations" -> run exactly stop_value generations
               "saturation"  -> stop after stop_value generations without improvement
    Returns: best individual, history (best fitness of every generation)
    """
    def fit(ind):
        return 1.0 / max(problem.cost(ind), 1e-9)      # shorter -> bigger fitness

    population = [problem.random_individual() for _ in range(pop_size)]
    best = max(population, key=fit)
    best_fitness = fit(best)

    # Three histories:
    # 1) current generation best fitness
    # 2) best fitness found so far
    # 3) average population fitness
    current_history = [best_fitness]
    best_history = [best_fitness]
    average_history = [sum(fit(p) for p in population) / len(population)]

    stall, gen = 0, 1

    while True:
        # ---- stopping criteria
        if stop_mode == "generations" and gen >= stop_value:
            break
        if stop_mode == "saturation" and (stall >= stop_value or gen >= MAX_GENERATIONS):
            break

        fits = [fit(p) for p in population]
        select = make_selector(population, fits, selection)

        #  elitism: copy the best individuals as they are
        ranked = sorted(population, key=fit, reverse=True)
        n_elite = int(pop_size * elit_rate)
        new_pop = [p[:] for p in ranked[:n_elite]]

        # fill the rest with crossover + mutation
        while len(new_pop) < pop_size:
            p1, p2 = select(), select()
            c1, c2 = p1[:], p2[:]
            if random.random() < cx_rate:
                c1, c2 = problem.crossover(p1, p2), problem.crossover(p2, p1)
            for child in (c1, c2):
                if random.random() < mut_rate:
                    child = problem.mutate(child)
                new_pop.append(child)
        population = new_pop[:pop_size]

        gen += 1

        # Fitness of the current generation
        fits_now = [fit(p) for p in population]
        gen_best = max(population, key=fit)
        gen_best_fitness = fit(gen_best)
        avg_fitness = sum(fits_now) / len(fits_now)

        # Keep the best solution found during the whole run
        if gen_best_fitness > best_fitness + 1e-12:
            best = gen_best[:]
            best_fitness = gen_best_fitness
            stall = 0
        else:
            stall += 1

        current_history.append(gen_best_fitness)
        best_history.append(best_fitness)
        average_history.append(avg_fitness)

    return best, current_history, best_history, average_history


#GUI
EXAMPLE = """A: B(4), C(2)
B: D(5), F(10)
C: F(3), B(1)
D: F(2)
F:"""


def fmt(x):
    return f"{x:g}"


def read_number(var, name, lo, hi, integer=False):
    try:
        v = int(var.get()) if integer else float(var.get())
    except ValueError:
        raise ValueError(f"{name} must be a valid {'whole ' if integer else ''}number.")
    if not lo <= v <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}.")
    return v


class App:
    def __init__(self, root):
        self.G = None
        self.pos = None
        self.graph_text = None
        root.title("Project 3 - Traveling and Shipment Routing (Genetic Algorithm)")
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        left = ttk.Frame(root, padding=10)
        left.grid(row=0, column=0, sticky="ns")
        right = ttk.Frame(root, padding=10)
        right.grid(row=0, column=1, sticky="nsew")

        # ---- graph description
        ttk.Label(left, text="Graph description (adjacency list or matrix)").pack(anchor="w")
        self.txt = tk.Text(left, width=36, height=8)
        self.txt.insert("1.0", EXAMPLE)
        self.txt.pack(pady=4)

        ttk.Button(left, text="Create Graph", command=self.create_graph).pack(pady=6)

        # ---- problem type
        pt = ttk.LabelFrame(left, text="Problem", padding=6)
        pt.pack(fill="x", pady=4)
        self.problem = tk.StringVar(value="path")
        ttk.Radiobutton(pt, text="Shortest path (Start -> End)",
                        variable=self.problem, value="path").pack(anchor="w")
        ttk.Radiobutton(pt, text="Traveling (visit all nodes)",
                        variable=self.problem, value="tour").pack(anchor="w")
        self.back = tk.BooleanVar(value=False)
        ttk.Checkbutton(pt, text="Return to Start (traveling only)",
                        variable=self.back).pack(anchor="w")
        se = ttk.Frame(pt)
        se.pack(anchor="w", pady=2)
        ttk.Label(se, text="Start:").grid(row=0, column=0)
        self.start_var = tk.StringVar(value="A")
        ttk.Entry(se, textvariable=self.start_var, width=6).grid(row=0, column=1, padx=4)
        ttk.Label(se, text="End:").grid(row=0, column=2)
        self.end_var = tk.StringVar(value="F")
        ttk.Entry(se, textvariable=self.end_var, width=6).grid(row=0, column=3, padx=4)

        # ---- stopping criteria
        sc = ttk.LabelFrame(left, text="Stopping criteria", padding=6)
        sc.pack(fill="x", pady=4)
        self.stop_mode = tk.StringVar(value="generations")
        ttk.Radiobutton(sc, text="Saturation (gens without improvement)",
                        variable=self.stop_mode, value="saturation").pack(anchor="w")
        ttk.Radiobutton(sc, text="After Generation",
                        variable=self.stop_mode, value="generations").pack(anchor="w")
        self.stop_value = tk.StringVar(value="50")
        ttk.Entry(sc, textvariable=self.stop_value, width=8).pack(anchor="w", pady=2)

        # ---- numeric parameters
        pf = ttk.Frame(left)
        pf.pack(fill="x", pady=4)
        self.pop = tk.StringVar(value="10")
        self.cx = tk.StringVar(value="80")
        self.mut = tk.StringVar(value="30")
        self.elit = tk.StringVar(value="10")
        for r, (label, var) in enumerate([("Population", self.pop), ("Crossover %", self.cx),
                                          ("Mutation %", self.mut), ("Elitism %", self.elit)]):
            ttk.Label(pf, text=label).grid(row=r, column=0, sticky="w", pady=2)
            ttk.Entry(pf, textvariable=var, width=8).grid(row=r, column=1, padx=8)

        # ---- selection
        sel = ttk.LabelFrame(left, text="Selection", padding=6)
        sel.pack(fill="x", pady=4)
        self.selection = tk.StringVar(value="Ranking")
        ttk.Radiobutton(sel, text="Ranking", variable=self.selection, value="Ranking").pack(anchor="w")
        ttk.Radiobutton(sel, text="Tournament", variable=self.selection, value="Tournament").pack(anchor="w")

        ttk.Button(left, text="Run Genetic Algorithm", command=self.run).pack(pady=8)
        self.result = ttk.Label(left, text="", wraplength=260, justify="left")
        self.result.pack(anchor="w")

        # ---- plots (graph on top, fitness curve at the bottom)
        self.fig_g = Figure(figsize=(5, 3.2))
        self.ax_g = self.fig_g.add_subplot(111)
        self.canvas_g = FigureCanvasTkAgg(self.fig_g, master=right)
        self.canvas_g.get_tk_widget().pack(fill="both", expand=True)

        self.fig_f = Figure(figsize=(5, 3.2))
        self.ax_f = self.fig_f.add_subplot(111)
        self.canvas_f = FigureCanvasTkAgg(self.fig_f, master=right)
        self.canvas_f.get_tk_widget().pack(fill="both", expand=True)

    def draw_graph(self, route=None):
        G, pos, ax = self.G, self.pos, self.ax_g
        ax.clear()
        size = 700
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#b0b0b0", node_size=size)
        nx.draw_networkx_labels(G, pos, ax=ax)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#888888",
                               arrows=True, arrowsize=18, node_size=size)
        labels = {(u, v): fmt(d["weight"]) for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax, font_size=8)
        if route:
            edges = [(a, b) for a, b in zip(route, route[1:]) if G.has_edge(a, b)]
            if edges:
                nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, edge_color="red",
                                       width=2.5, arrows=True, arrowsize=20, node_size=size)
        ax.set_axis_off()
        self.canvas_g.draw()

    def create_graph(self):
        text = self.txt.get("1.0", "end").strip()
        try:
            G = parse_graph(text)
        except ValueError as e:
            messagebox.showerror("Graph error", str(e))
            return False
        self.G, self.graph_text = G, text
        self.pos = nx.spring_layout(G, seed=7)
        nodes = list(G.nodes)
        if self.start_var.get().strip() not in G:      # auto-fill Start / End
            self.start_var.set(nodes[0])
        if self.end_var.get().strip() not in G:
            self.end_var.set(nodes[-1])
        self.draw_graph()
        return True

    def run(self):
        text = self.txt.get("1.0", "end").strip()
        if self.G is None or text != self.graph_text:   # text was edited -> rebuild
            if not self.create_graph():
                return
        try:
            start = self.start_var.get().strip()
            end = self.end_var.get().strip()
            if start not in self.G:
                raise ValueError(f"Start node '{start}' is not in the graph.")
            pop = read_number(self.pop, "Population", 2, 10000, integer=True)
            cx = read_number(self.cx, "Crossover %", 0, 100) / 100
            mut = read_number(self.mut, "Mutation %", 0, 100) / 100
            elit = read_number(self.elit, "Elitism %", 0, 100) / 100
            stop_value = read_number(self.stop_value, "Stopping value", 1, 100000, integer=True)

            if self.problem.get() == "path":
                if end not in self.G or end == start:
                    raise ValueError("End must be a node of the graph, different from Start.")
                problem = PathProblem(self.G, start, end)
            else:
                problem = TourProblem(self.G, start, self.back.get())

            best, current_history, best_history, average_history = run_ga(
                problem, pop, cx, mut, elit,
                self.selection.get(), self.stop_mode.get(), stop_value)
        except ValueError as e:
            messagebox.showerror("Input error", str(e))
            return

        route = problem.route(best)
        if problem.is_valid(best):
            info = f"Best route: {' -> '.join(route)}\nCost: {fmt(problem.cost(best))}"
        else:
            info = ("No valid route found: some required edges do not exist in the graph.\n"
                    f"Best attempt: {' -> '.join(route)}")
        self.result.config(
            text=f"{info}\nGenerations: {len(best_history)}"
                 f"\nBest fitness: {best_history[-1]:.6f}")
        self.draw_graph(route)

        ax = self.ax_f
        ax.clear()
        generations = range(1, len(best_history) + 1)

        # Current generation best can go up and down.
        ax.plot(generations, current_history,
                marker="o", markersize=3, label="Current Generation Best")

        # Best-so-far can only stay the same or improve.
        ax.plot(generations, best_history,
                linewidth=2, label="Best Fitness So Far")

        # Average fitness shows how the whole population changes.
        ax.plot(generations, average_history,
                linestyle="--", alpha=0.8, label="Average Fitness")

        ax.set_title("Genetic Algorithm Fitness")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness (1 / cost)")
        ax.grid(alpha=0.3)
        ax.legend()
        self.fig_f.tight_layout()
        self.canvas_f.draw()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()