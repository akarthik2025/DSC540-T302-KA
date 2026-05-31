"""
Milestone 5: Merge Data, Store in Database, and Visualize
Name: Karthikeya Allada
Date: 05/30/2026
Project: Global Quality of Life Analysis

This script is a Python-file version of the Milestone5_Final notebook.
It loads cleaned data from three sources (flat file, website, API), stores
all sources in SQLite, joins them into one dataset, prints key outputs,
and generates five labeled visualizations.
"""

from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns


def resolve_paths() -> tuple[Path, Path, Path, Path]:
    """Resolve project paths whether this script runs from Project or MileStone5 folder."""
    cwd = Path.cwd()
    project_root = cwd if (cwd / "MileStone2").exists() else cwd.parent

    m2_file = project_root / "MileStone2" / "world_happiness_cleaned_milestone2.csv"
    m3_file = project_root / "Milestone3" / "Milestone3_GDP_Cleaned_Final.csv"
    db_file = project_root / "MileStone5" / "term_project.db"
    merged_json = project_root / "MileStone5" / "milestone5_final_merged_output.json"

    return m2_file, m3_file, db_file, merged_json


def load_api_data() -> pd.DataFrame:
    """Pull API data directly and prepare country-level fields."""
    url = (
        "https://restcountries.com/v3.1/all?"
        "fields=name,capital,region,subregion,population,area,languages,"
        "currencies,latlng,timezones"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    raw_json = response.json()

    rows = []
    for country in raw_json:
        name_info = country.get("name", {})
        common_name = name_info.get("common", "")

        capitals = country.get("capital", [])
        capital = capitals[0] if capitals else "Unknown"

        region = country.get("region", "")
        subregion = country.get("subregion", "")
        population = country.get("population", np.nan)
        area = country.get("area", np.nan)

        latlng = country.get("latlng", [np.nan, np.nan])
        latitude = latlng[0] if len(latlng) > 0 else np.nan
        longitude = latlng[1] if len(latlng) > 1 else np.nan

        rows.append(
            {
                "country": common_name,
                "capital_city": capital,
                "continent": region,
                "subregion": subregion,
                "population": population,
                "area_sq_km": area,
                "latitude": latitude,
                "longitude": longitude,
            }
        )

    df_api = pd.DataFrame(rows)
    df_api["country"] = df_api["country"].str.strip().str.title()
    df_api["pop_density_per_sq_km"] = np.where(
        df_api["area_sq_km"] > 0,
        (df_api["population"] / df_api["area_sq_km"]).round(2),
        np.nan,
    )
    return df_api


def standardize_country_keys(
    df_happy: pd.DataFrame, df_gdp: pd.DataFrame, df_api: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Standardize country keys used for joins across all datasets."""
    df_happy = df_happy.rename(columns={"country": "country"})
    df_happy["country"] = df_happy["country"].astype(str).str.strip().str.title()

    df_gdp = df_gdp.rename(columns={"Country_Territory": "country"})
    df_gdp["country"] = df_gdp["country"].astype(str).str.strip().str.title()

    manual_key_fixes = {
        "United States Of America": "United States",
        "Czechia": "Czech Republic",
        "Türkiye": "Turkey",
        "South Korea": "Korea, Republic Of",
    }

    for frame in (df_happy, df_gdp, df_api):
        frame["country"] = frame["country"].replace(manual_key_fixes)

    return df_happy, df_gdp, df_api


def load_tables_to_sqlite(
    db_file: Path, df_happy: pd.DataFrame, df_gdp: pd.DataFrame, df_api: pd.DataFrame
) -> pd.DataFrame:
    """Persist source datasets as SQLite tables and return per-table row counts."""
    db_file.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_file) as conn:
        df_happy.to_sql("happiness_clean", conn, index=False, if_exists="replace")
        df_gdp.to_sql("gdp_clean", conn, index=False, if_exists="replace")
        df_api.to_sql("api_clean", conn, index=False, if_exists="replace")

        table_counts = pd.read_sql_query(
            """
            SELECT 'happiness_clean' AS table_name, COUNT(*) AS rows FROM happiness_clean
            UNION ALL
            SELECT 'gdp_clean' AS table_name, COUNT(*) AS rows FROM gdp_clean
            UNION ALL
            SELECT 'api_clean' AS table_name, COUNT(*) AS rows FROM api_clean
            """,
            conn,
        )

    return table_counts


def build_consolidated_dataset(db_file: Path) -> pd.DataFrame:
    """Join all three tables into one consolidated dataset and store it in SQLite."""
    join_sql = """
    SELECT
        h.country,
        h.region AS happiness_region,
        h.year,
        h.happiness_score,
        h.log_gdp_per_capita,
        h.social_support,
        h.healthy_life_expectancy,
        h.freedom_score,
        g.GDP_Billion_USD,
        g.Economy_Category,
        a.continent,
        a.subregion,
        a.population,
        a.area_sq_km,
        a.pop_density_per_sq_km
    FROM happiness_clean h
    LEFT JOIN gdp_clean g
        ON h.country = g.country
    LEFT JOIN api_clean a
        ON h.country = a.country
    """

    with sqlite3.connect(db_file) as conn:
        consolidated = pd.read_sql_query(join_sql, conn)
        consolidated.to_sql(
            "consolidated_quality_of_life", conn, index=False, if_exists="replace"
        )

    return consolidated


def generate_visualizations(consolidated: pd.DataFrame, output_dir: Path) -> None:
    """Create and save all five milestone visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualization 1: Top 15 countries by happiness score
    plot1 = (
        consolidated[["country", "happiness_score"]]
        .dropna()
        .sort_values("happiness_score", ascending=False)
        .head(15)
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot1, x="happiness_score", y="country", color="#4C78A8")
    plt.title("Visualization 1: Top 15 Countries by Happiness Score")
    plt.xlabel("Happiness Score")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.savefig(output_dir / "viz1_top15_happiness.png", dpi=300)
    plt.show()

    # Visualization 2: Top 15 countries by GDP
    plot2 = (
        consolidated[["country", "GDP_Billion_USD"]]
        .dropna()
        .sort_values("GDP_Billion_USD", ascending=False)
        .head(15)
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot2, x="GDP_Billion_USD", y="country", color="#59A14F")
    plt.title("Visualization 2: Top 15 Countries by GDP (Billion USD)")
    plt.xlabel("GDP (Billion USD)")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.savefig(output_dir / "viz2_top15_gdp.png", dpi=300)
    plt.show()

    # Visualization 3: Top 15 countries by population
    plot3 = (
        consolidated[["country", "population"]]
        .dropna()
        .sort_values("population", ascending=False)
        .head(15)
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot3, x="population", y="country", color="#F28E2B")
    plt.title("Visualization 3: Top 15 Countries by Population")
    plt.xlabel("Population")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.savefig(output_dir / "viz3_top15_population.png", dpi=300)
    plt.show()

    # Visualization 4: Happiness vs GDP with continent hue
    plot4 = consolidated[["country", "happiness_score", "GDP_Billion_USD", "continent"]].dropna()
    plt.figure(figsize=(11, 7))
    sns.scatterplot(
        data=plot4,
        x="GDP_Billion_USD",
        y="happiness_score",
        hue="continent",
        alpha=0.75,
    )
    plt.title("Visualization 4: Happiness vs GDP (Joined Happiness + GDP + API)")
    plt.xlabel("GDP (Billion USD)")
    plt.ylabel("Happiness Score")
    plt.tight_layout()
    plt.savefig(output_dir / "viz4_happiness_vs_gdp_continent.png", dpi=300)
    plt.show()

    # Visualization 5: Regional averages from merged tables
    plot5 = (
        consolidated.groupby("continent", as_index=False)
        .agg(
            {
                "happiness_score": "mean",
                "GDP_Billion_USD": "mean",
                "pop_density_per_sq_km": "mean",
            }
        )
        .dropna()
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    sns.barplot(
        data=plot5, x="continent", y="happiness_score", ax=axes[0], color="#9C6ADE"
    )
    axes[0].set_title("Avg Happiness by Continent")
    axes[0].tick_params(axis="x", rotation=35)

    sns.barplot(
        data=plot5, x="continent", y="GDP_Billion_USD", ax=axes[1], color="#7F7F7F"
    )
    axes[1].set_title("Avg GDP (Billion USD) by Continent")
    axes[1].tick_params(axis="x", rotation=35)

    sns.barplot(
        data=plot5,
        x="continent",
        y="pop_density_per_sq_km",
        ax=axes[2],
        color="#E15759",
    )
    axes[2].set_title("Avg Population Density by Continent")
    axes[2].tick_params(axis="x", rotation=35)

    fig.suptitle("Visualization 5: Continent-level Metrics from Joined Data", y=1.03)
    plt.tight_layout()
    plt.savefig(output_dir / "viz5_continent_level_metrics.png", dpi=300)
    plt.show()


def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    sns.set_theme(style="whitegrid")

    m2_file, m3_file, db_file, merged_json = resolve_paths()

    print("Paths loaded:")
    print(f"  Milestone 2 file: {m2_file}")
    print(f"  Milestone 3 file: {m3_file}")
    print(f"  SQLite DB: {db_file}")

    # Step 2: Load cleaned flat-file and website datasets
    df_happy = pd.read_csv(m2_file)
    df_gdp = pd.read_csv(m3_file)

    print(f"World Happiness rows x cols: {df_happy.shape}")
    print(f"GDP rows x cols: {df_gdp.shape}")
    print("World Happiness preview:")
    print(df_happy.head(3))
    print("GDP preview:")
    print(df_gdp.head(3))

    # Step 3: Pull API data
    df_api = load_api_data()
    print(f"API rows x cols: {df_api.shape}")
    print("API preview:")
    print(df_api.head(3))

    # Step 4: Standardize keys
    df_happy, df_gdp, df_api = standardize_country_keys(df_happy, df_gdp, df_api)
    print("Join key standardization complete.")
    print(f"Unique countries - happiness: {df_happy['country'].nunique()}")
    print(f"Unique countries - gdp: {df_gdp['country'].nunique()}")
    print(f"Unique countries - api: {df_api['country'].nunique()}")

    # Step 5: Write source tables
    table_counts = load_tables_to_sqlite(db_file, df_happy, df_gdp, df_api)
    print("Tables loaded into SQLite:")
    print(table_counts)

    # Step 6: Build consolidated table
    consolidated = build_consolidated_dataset(db_file)
    print(f"Consolidated dataset rows x cols: {consolidated.shape}")
    print("Null overview for key merged columns:")
    print(consolidated[["GDP_Billion_USD", "population", "pop_density_per_sq_km"]].isna().sum())
    print("Consolidated preview:")
    print(consolidated.head(10))

    # Final merged dataset output
    print(
        f"Final Merged Dataset: {consolidated.shape[0]} rows x {consolidated.shape[1]} columns"
    )
    final_preview = consolidated.sort_values(
        ["happiness_score", "GDP_Billion_USD"], ascending=[False, False]
    ).head(25)
    print(final_preview)

    # Save merged output as JSON for reproducible artifact
    consolidated.to_json(merged_json, orient="records", indent=2)
    print(f"Merged dataset JSON saved: {merged_json}")

    # Generate and save all visualizations
    generate_visualizations(consolidated, db_file.parent)


if __name__ == "__main__":
    main()
