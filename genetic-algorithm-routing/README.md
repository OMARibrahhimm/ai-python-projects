# Traveling & Shipment Routing with a Genetic Algorithm

A Python/Tkinter app that solves two routing problems on a weighted directed graph using a genetic algorithm:

- **Shortest path** from a start node to an end node.
- **Traveling** (visit every node), with an optional return to the start.

## Features
- Graph input as an **adjacency list** or an **adjacency matrix**, with validation and clear error messages.
- Two selection methods: **Ranking** and **Tournament**.
- Configurable population size, crossover %, mutation %, and elitism %.
- Two stopping criteria: fixed number of generations, or **saturation** (generations without improvement).
- Custom operators per problem: path-splicing crossover for paths, **order crossover** + swap mutation for tours.
- Live plots: the graph with the best route highlighted, and a fitness chart showing the current-generation best, the best so far, and the population average.

## Run
```bash
pip install -r ../requirements.txt
python ga_routing.py
```

## Graph format
```
A: B(4), C(2)
B: D(5), F(10)
C: F(3), B(1)
D: F(2)
F:
```
Weight goes in brackets (default 1). A matrix works too: first line = node names, then one row per node, `0` = no edge.

## How it works
Fitness is `1 / cost`, so shorter routes score higher. Each generation keeps the elite individuals, picks parents with the chosen selection method, then applies crossover and mutation to fill the next population.
