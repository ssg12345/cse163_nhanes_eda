

# Sahana Ganesan
# CSE 163 AB - Autumn 2026
# EDA: Lifestyle Factors and Cognitive Performance
# This file loads NHANES data and looks at how age, sleep, and physical
# activity relate to cognitive performance.

import pandas as pd
import matplotlib.pyplot as plt


def clean_data(data):
    """
    Keeps adults 60 and older and removes missing or invalid responses.
    """
    older_adults = data[data["RIDAGEYR"] >= 60].copy()
    older_adults["SLD010H"] = older_adults["SLD010H"].replace(
        {77: float("nan"), 99: float("nan")}
    )
    return older_adults.dropna(
        subset=["SLD010H", "CFDDS", "PAQ650"]
    )


def summarize_data(data):
    """
    Prints the dataset size, missing values, and summary statistics.
    """
    columns = ["RIDAGEYR", "SLD010H", "CFDDS"]

    print("Dataset size:", data.shape)
    print("Missing values:")
    print(data[columns + ["PAQ650"]].isna().sum())
    print("Numerical summaries:")
    print(data[columns].describe())
    print("Physical activity counts:")
    print(data["PAQ650"].value_counts(dropna=False).sort_index())


def make_graphs(data):
    """
    Makes graphs comparing age, sleep, and physical activity with
    cognitive scores, and saves them as images.
    """
    plt.figure()
    plt.scatter(data["RIDAGEYR"], data["CFDDS"], alpha=0.3)
    plt.xlabel("Age (years)")
    plt.ylabel("Cognitive score")
    plt.title("Age and Cognitive Performance")
    plt.tight_layout()
    plt.savefig("age_vs_cognition.png")
    plt.close()

    plt.figure()
    plt.scatter(data["SLD010H"], data["CFDDS"], alpha=0.3)
    plt.xlabel("Sleep duration (hours)")
    plt.ylabel("Cognitive score")
    plt.title("Sleep Duration and Cognitive Performance")
    plt.tight_layout()
    plt.savefig("sleep_vs_cognition.png")
    plt.close()

    activity_means = data.groupby("PAQ650")["CFDDS"].mean()

    plt.figure()
    plt.bar(["Active", "Inactive"],
            [activity_means[1.0], activity_means[2.0]])
    plt.xlabel("Vigorous recreational activity")
    plt.ylabel("Average cognitive score")
    plt.title("Physical Activity and Cognitive Performance")
    plt.tight_layout()
    plt.savefig("activity_vs_cognition.png")
    plt.close()

    sleep_means = data.groupby("SLD010H")["CFDDS"].mean()

    plt.figure()
    plt.plot(sleep_means.index, sleep_means.values, marker="o")
    plt.xlabel("Sleep duration (hours)")
    plt.ylabel("Average cognitive score")
    plt.title("Average Cognitive Score by Sleep Duration")
    plt.tight_layout()
    plt.savefig("sleep_averages.png")
    plt.close()

    age_means = data.groupby("RIDAGEYR")["CFDDS"].mean()

    plt.figure()
    plt.plot(age_means.index, age_means.values)
    plt.xlabel("Age (years)")
    plt.ylabel("Average cognitive score")
    plt.title("Average Cognitive Score by Age")
    plt.tight_layout()
    plt.savefig("age_averages.png")
    plt.close()

    print("Average cognitive scores by activity:")
    print(activity_means)


def check_relationships(data):
    """
    Checks correlations and compares cognitive scores between
    the physical activity groups.
    """
    print("Correlations with cognitive score:")
    print(data[["RIDAGEYR", "SLD010H", "CFDDS"]].corr()["CFDDS"])

    print("Cognitive score summary by activity:")
    print(data.groupby("PAQ650")["CFDDS"].describe())

    print("Number of participants by activity:")
    print(data["PAQ650"].value_counts())

    print("Cognitive score ranges:")
    print(data["CFDDS"].min(), data["CFDDS"].max())

    print("Sleep duration ranges:")
    print(data["SLD010H"].min(), data["SLD010H"].max())


def check_assumptions(data):
    """
    Checks for potential outliers, duplicate IDs, and differences
    in cognitive scores between activity groups.
    """
    print("Cognitive score distributions by activity:")
    print(data.groupby("PAQ650")["CFDDS"].describe())

    print("Potential outliers:")
    for column in ["RIDAGEYR", "SLD010H", "CFDDS"]:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = ((data[column] < lower) |
                 (data[column] > upper)).sum()

        print(column, ":", count)

    print("Duplicate participant IDs:")
    print(data["SEQN"].duplicated().sum())


def main():
    """
    Loads and combines the NHANES datasets, cleans the data,
    and runs the summaries, graphs, and assumption checks.
    """
    cognitive = pd.read_sas("CFQ_H.xpt")
    sleep = pd.read_sas("SLQ_H.xpt")
    activity = pd.read_sas("PAQ_H.xpt")
    demographics = pd.read_sas("DEMO_H.xpt")

    print("Cognitive:", cognitive.shape)
    print("Sleep:", sleep.shape)
    print("Activity:", activity.shape)
    print("Demographics:", demographics.shape)

    print("Cognitive summary:")
    print(cognitive["CFDDS"].describe())

    print("Sleep summary:")
    print(sleep["SLD010H"].value_counts(dropna=False).sort_index())

    print("Activity summary:")
    print(activity["PAQ650"].value_counts(dropna=False).sort_index())

    print("Age summary:")
    print(demographics["RIDAGEYR"].describe())

    merged = cognitive.merge(sleep, on="SEQN", how="inner")
    merged = merged.merge(activity, on="SEQN", how="inner")
    merged = merged.merge(demographics, on="SEQN", how="inner")

    print("Merged dataset:", merged.shape)

    older_adults = merged[merged["RIDAGEYR"] >= 60].copy()

    print("Adults 60+:", older_adults.shape)
    print("Missing cognitive scores:",
          older_adults["CFDDS"].isna().sum())

    print("Missing values before cleaning:")
    print(older_adults[
        ["RIDAGEYR", "SLD010H", "PAQ650", "CFDDS"]
    ].isna().sum())

    print("Sleep duration values before cleaning:")
    print(older_adults["SLD010H"].value_counts(
        dropna=False).sort_index())

    clean = clean_data(merged)

    print("Clean dataset:")
    summarize_data(clean)

    make_graphs(clean)
    check_relationships(clean)
    check_assumptions(clean)
    print("All graphs saved successfully.")


if __name__ == "__main__":
    main()
