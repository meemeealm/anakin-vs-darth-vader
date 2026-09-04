# The Anakin Skywalker T-GNN : Tracking a Character's Fall with AI

I was wondering if machine learning can track a fictional character's emotional breakdown.Instead of building another predictor, I wanted to do something, a model to map Anakin Skywalker's psychological descent to the dark side across 16 different eras of his life.

## What It Does

This project uses a **Temporal Graph Neural Network (T-GNN)**. It is such an AI that understands relationships (like who Anakin is hanging out with) and remembers how things change over time.

At every era in his timeline, the model looks at:

* **Graph connections:** Who he is interacting with (Jedi, clones, Palpatine).
* **Stress levels:** His changing mental load.
* **Palpatine's proximity:** How close he is to bad influences.

By feeding this data through the network step-by-step, the model learns to classify his mental state (from "Optimistic" to "Fallen") as his story progresses.

$$h^{(t)} = \text{GRUCell}\left(\text{GCN}(X^{(t)}, E^{(t)}), h^{(t-1)}\right)$$

Explanation: at every era snapshot ($t$), the model takes Anakin's current feature state ($X^{(t)}$) and his social network connections ($E^{(t)}$), blends them spatially using graph convolution, and passes that output into the recurrent cell alongside his historical psychological memory ($h^{(t-1)}$) to compute his updated mental state ($h^{(t)}$).

## Black Box

Machine learning models are usually total mysteries, but I included tools here to see *how* it thinks:

* **PCA Trajectory Maps:** I took the model's hidden memory and squashed it down into a 2D map. When you connect the dots across time, it doesn't just draw a straight line—it forms loops and sharp turns, showing visually when Anakin's mental state circled back to old habits or hit a major point.
* **Feature Saliency (Gradients):** By checking what the model paid attention to, I found out it cared way more about his baseline identity and how close he was to Palpatine than it did about raw stress scores during critical moments.

## Tech Stack

* Python
* PyTorch & PyTorch Geometric (for the graph neural network)
* Scikit-learn (for PCA and metrics)
* Matplotlib