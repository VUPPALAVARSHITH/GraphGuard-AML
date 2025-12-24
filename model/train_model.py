"""
train_model.py

Training utilities for GraphSAGE + GAT AML model.
"""

import torch


def train_one_epoch(model, edge_index, edge_attr, labels, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    logits = model(edge_index, edge_attr)
    loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()

    return loss.item()
