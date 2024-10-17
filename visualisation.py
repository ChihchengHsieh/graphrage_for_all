import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt

# Plot the density of "Augmented_Body Temperature (°C)" using seaborn

def main():
    plt.figure(figsize=(10, 6))
    df_with_db = pd.read_csv("combined_augmented_all_llama3_1.csv")
    sns.kdeplot(df_with_db["Augmented_Body Temperature (°C)"], label="With GraphRAG")
    plt.title("Density Plot of Augmented Body Temperature (°C)")
    plt.xlabel("Augmented Body Temperature (°C)")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("./result_examples/density_plot_augmented_body_temperature.png")
    time.sleep(5)


if __name__ == "__main__":
    main()