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

---

## How to Read the Charts

 The Streamlit dashboard shows how Anakin's psychological state changes across different stages of his story. Each chart looks at the character from a different perspective.

 | Chart | What it represents | How to read it |
| --- | --- | --- |
| **Social Graph** | Shows Anakin's relationships with other characters at a selected era. | Each circle is a character. Lines represent relationships. Purple represents Palpatine. Orange/red represents Anakin/Vader. |
| **Mental-State Trajectory (PCA)** | Shows how the T-GNN represents Anakin/Vader's mental states internally. | Each point represents a mental state such as **Optimistic**, **Conflicted**, **Desperate/Traumatized**, **Numb/Cold**, or **At Peace**. |
| **Principal Component 1 (PC1)** | The main direction of variation in the model's hidden representations. | Moving left or right means the model's representation changes along this direction. PC1 does not directly mean stress or time. |
| **Principal Component 2 (PC2)** | The second major direction of variation in the hidden representations. | Moving up or down means the representation changes along another direction. PC2 does not directly represent a specific psychological variable. |
| **Red Circle** | Shows the currently selected era on the PCA chart. | Changing the era in the sidebar moves the red circle to that era's mental state. |
| **Model Probabilities** | Shows the model's confidence in each of the five psychological state classes. | A taller bar means the model gives that state a higher probability. |
| **Predicted State** | The psychological state class selected by the model. | The model chooses the class with the highest probability. |
| **Stress Score** | Represents Anakin/Vader's stress level used as a model input. | A value closer to **1.0** means higher stress. A value closer to **0.0** means lower stress. |
| **Palpatine Proximity** | Represents Anakin/Vader's level of connection or influence from Palpatine. | A value closer to **1.0** means stronger proximity or influence. |

 ## Understanding the PCA Chart

 The PCA chart is **not a timeline chart**. The x-axis and y-axis do not directly represent stress, time, or a specific psychological measurement.

 Instead, the T-GNN creates a **hidden representation** of Anakin/Vader at each era. PCA reduces these high-dimensional hidden representations into two dimensions so they can be visualized.

 The points represent the progression:

 **Optimistic → Conflicted → Desperate/Traumatized → Numb/Cold → At Peace**

 The line connects the points in chronological order.

 Points that are close together have more similar representations inside the model. Points that are farther apart have more different representations.

 ## Reading the Social Graph

 The Social Graph shows the character network at the selected era.

 - **Nodes** represent characters.
- **Edges** represent relationships between characters.
- **Purple** represents Palpatine.
- **Orange/red** represents Anakin/Vader.
- **Blue** represents other characters.
- A changing network structure shows how Anakin's social environment changes across the story.

 ## Reading the Model Probabilities

 The probability chart shows the model's prediction across five state classes.

 For example:

```
State 0  ████████████████  0.80
State 1  ███               0.10
State 2  ██                0.06
State 3  █                 0.03
State 4  █                 0.01
```

 In this example, the model predicts **State 0** because it has the highest probability.

 The probabilities should be read as the model's confidence, not as a guaranteed psychological diagnosis.

 ## Using the "What If?" Controls

 The sidebar allows you to experiment with different conditions.

 ### Era Snapshot

 Selects which era you want to examine. The selected era is highlighted on the PCA chart.

 ### Stress Score

 Changes Anakin/Vader's stress input for the selected era.

 - **0.0** = low stress
- **1.0** = high stress

 ### Palpatine Proximity

 Changes Anakin/Vader's Palpatine influence input for the selected era.

 - **0.0** = low proximity/influence
- **1.0** = high proximity/influence

 After changing these values, the model produces a new prediction.

 ## Important Note

 The PCA visualization shows how the **trained T-GNN represents Anakin/Vader's mental states internally**.

 It does **not** prove that two mental states are psychologically similar in the real world, and it is not a clinical psychological measurement.

 The PCA should therefore be interpreted as a visualization of the **model's learned representation**.