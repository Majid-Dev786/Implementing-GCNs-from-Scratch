# I'm starting off by importing all the necessary libraries and modules. 
# This includes numpy for numerical operations,torch and its submodules for neural network components, networkx for graph manipulation, and matplotlib for plotting.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Here, I define the GraphConvolution class, which is a basic building block for creating graph convolutional layers.
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        # I initiate the class by inheriting from nn.Module and setting up a linear transformation.
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # The forward pass multiplies the adjacency matrix by the input features before applying the linear transformation.
        x = torch.matmul(adj, x)
        x = self.linear(x)
        return x

# Here, I define a neural network model that uses the graph convolutional layers defined above.
class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, n_features, n_classes):
        # The network has two graph convolutional layers and is initialized here.
        super(GraphConvolutionalNetwork, self).__init__()
        self.conv1 = GraphConvolution(n_features, 16)
        self.conv2 = GraphConvolution(16, n_classes)

    def forward(self, x, adj):
        # In the forward pass, I apply the first graph convolution, followed by a ReLU and dropout, before the final graph convolution.
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.conv2(x, adj)
        return x

# This function generates synthetic data to simulate a graph with features and labels for nodes.
def generate_synthetic_data(num_nodes=50):
    # I create a random geometric graph, convert its adjacency matrix to a tensor, and randomly generate features and labels.
    G = nx.random_geometric_graph(num_nodes, radius=0.2)
    adj = nx.to_numpy_array(G)
    adj = torch.tensor(adj, dtype=torch.float32)

    features = np.random.rand(num_nodes, 2)
    features = torch.tensor(features, dtype=torch.float32)

    labels = np.random.randint(0, 2, num_nodes)
    labels = torch.tensor(labels, dtype=torch.long)
    return G, features, adj, labels

# This function trains the graph convolutional network model using the generated synthetic data.
def train_model(model, features, adj, labels, epochs=100):
    # I set up the optimizer and the loss function here. The training loop updates the model's weights based on the loss.
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # Each epoch, I perform a forward pass, compute the loss, and update the model's weights.
        model.train()
        optimizer.zero_grad()
        logits = model(features, adj)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            # Every 10 epochs, I print the current loss to monitor the training progress.
            print(f"Epoch {epoch}, Loss: {loss.item()}")

# This function visualizes the graph, its nodes colored by their predicted labels.
def visualize_graph(G, labels, pred_labels, title="Graph Convolutional Network"):
    # I use networkx to draw the graph and matplotlib to display it.
    pos = nx.get_node_attributes(G, "pos")
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, node_size=300, node_color=pred_labels, cmap=plt.cm.RdYlBu, with_labels=True)
    plt.title(title)
    plt.show()

# This is the main block where everything comes together.
if __name__ == "__main__":
    # I generate synthetic data and initialize the model.
    G, features, adj, labels = generate_synthetic_data()

    # Next, I instantiate the model, train it with the synthetic data, and evaluate it.
    gcn = GraphConvolutionalNetwork(n_features=2, n_classes=2)
    train_model(gcn, features, adj, labels)

    # After training, I switch the model to evaluation mode and make predictions on the data.
    gcn.eval()
    with torch.no_grad():
        logits = gcn(features, adj)
        pred_labels = torch.argmax(logits, dim=1).numpy()

    # Finally, I visualize the graph with the predicted labels.
    visualize_graph(G, labels, pred_labels)
