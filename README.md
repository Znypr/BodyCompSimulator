# Body Composition Dashboard 📊

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-3399FF)
![License](https://img.shields.io/badge/License-MIT-green)

A scientific body composition simulator built with Python. Unlike simple linear weight trackers, this dashboard uses physiological constraints—such as **Forbes Law** and the **Alpert Limit**—to model realistic muscle gain and fat loss trajectories over multi-year bulk/cut cycles.

![Dashboard Screenshot](imgs/screenshot.png)

## 🚀 Key Features

### 🧬 Physiological Modeling
* **Forbes Curve Integration:** Dynamically adjusts your "P-Ratio" (partitioning ratio). The simulator understands that leaner individuals gain more muscle in a surplus, while those with higher body fat gain preferentially more fat.
* **Smart Muscle Protection (Alpert Limit):** Enforces the biological limit of fat oxidation (~69 kcal/kg of fat mass/day). If you attempt a deficit larger than your fat stores can support, the simulation warns you and models the resulting muscle catabolism.
* **Metabolic Staleness:** Simulates "diet fatigue." Anabolic efficiency decays the longer you stay in a surplus, encouraging realistic cycle structuring.

### 🛠 Advanced Protocol Planning
* **Geometric Cycle Scaling:** Automatically scale the length of future bulk/cut cycles to model the increasing difficulty of gains over time.
* **Custom Priming Phases:** Configure a specific "Mini-Cut" or "Kickstart Bulk" before settling into your long-term cycle rhythm.
* **Interactive Visualizations:**
    * Combined Dual-Axis projections (Weight vs. Body Fat %).
    * Tissue Composition charts (Lean Mass vs. Fat Mass).
    * Visual "Efficiency Zones" based on your starting body fat.

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/Znypr/BodyCompSimulator.git](https://github.com/Znypr/BodyCompSimulator.git)
    cd BodyCompSimulator
    ```

2.  **Install dependencies**
    ```bash
    pip install streamlit plotly numpy pandas
    ```

3.  **Run the application**
    ```bash
    streamlit run main.py
    ```

## 🧠 The Science Behind It

### 1. Forbes Law (Partitioning Ratio)
The amount of muscle you gain in a surplus is mathematically linked to your current body fat percentage.
> *Formula:* `Lean Gain Ratio = 10.4 / (10.4 + Fat Mass)`

This means a user starting at 10% body fat will partition significantly more calories into muscle than a user starting at 20% body fat.

### 2. The Alpert Limit (Muscle Sparing)
Research suggests the human body can mobilize approximately **69 kcal per kilogram of fat mass per day**.
* **Safe Zone:** If your deficit < (Fat Mass * 69), you lose mostly fat.
* **Danger Zone:** If your deficit > (Fat Mass * 69), the remaining energy must be created by breaking down lean muscle tissue.

This dashboard visualizes exactly when you cross this threshold (displayed as "Unsafe" red zones in the charts).

## 🎨 Tech Stack
* **Frontend:** [Streamlit](https://streamlit.io/) for the interactive UI.
* **Visualization:** [Plotly Graph Objects](https://plotly.com/python/) for high-performance, interactive vector charts.
* **Logic:** [NumPy](https://numpy.org/) & [Pandas](https://pandas.pydata.org/) for vectorized simulation dataframes.

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
