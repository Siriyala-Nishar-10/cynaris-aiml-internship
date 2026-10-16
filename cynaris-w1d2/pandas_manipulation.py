"""
W1D2: Pandas for Data Manipulation
Task: load a real Indian dataset, perform filter/groupby/merge/pivot_table,
and export cleaned data to CSV and Parquet.

Author: Siriyala Nishar
"""

from pathlib import Path
import pandas as pd


def load_and_inspect(csv_path: Path) -> pd.DataFrame:
    """Load the CSV and print shape, dtypes, and first 10 rows."""
    df = pd.read_csv(csv_path)
    print("Shape:", df.shape)
    print("\nDtypes:\n", df.dtypes)
    print("\nHead(10):\n", df.head(10))
    return df


def filter_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Filter: keep only high-value orders (total_amount > 5000)."""
    filtered = df[df["total_amount"] > 5000]
    print(f"\nFiltered {len(filtered)} of {len(df)} orders with total > 5000")
    return filtered


def groupby_city_category(df: pd.DataFrame) -> pd.DataFrame:
    """Groupby: total revenue per city and category combination."""
    grouped = (
        df.groupby(["city", "category"])["total_amount"]
        .sum()
        .reset_index()
        .sort_values("total_amount", ascending=False)
    )
    print("\nTop 5 city-category revenue combos:\n", grouped.head())
    return grouped


def merge_with_city_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Merge: attach a small city-level metadata table (region) via merge."""
    city_region = pd.DataFrame({
        "city": ["Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Pune"],
        "region": ["South", "West", "North", "South", "South", "West"],
    })
    # left merge keeps every order row, adds region column where city matches
    merged = df.merge(city_region, on="city", how="left")
    print("\nMerged sample:\n", merged[["city", "region", "total_amount"]].head())
    return merged


def pivot_revenue_by_region_category(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot table: revenue by region (rows) vs category (columns)."""
    pivot = pd.pivot_table(
        df, values="total_amount", index="region", columns="category",
        aggfunc="sum", fill_value=0
    )
    print("\nPivot table (region x category revenue):\n", pivot)
    return pivot


def export_and_compare(df: pd.DataFrame, base_name: str) -> None:
    """Export cleaned DataFrame to CSV and Parquet, compare file sizes."""
    csv_path = Path(f"{base_name}.csv")
    parquet_path = Path(f"{base_name}.parquet")

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    csv_size = csv_path.stat().st_size
    parquet_size = parquet_path.stat().st_size

    print(f"\nCSV size: {csv_size:,} bytes")
    print(f"Parquet size: {parquet_size:,} bytes")
    print(f"Parquet is {csv_size / parquet_size:.2f}x smaller than CSV")


def main() -> None:
    csv_path = Path("indian_retail_orders.csv")

    df = load_and_inspect(csv_path)
    filter_orders(df)
    groupby_city_category(df)
    merged = merge_with_city_metadata(df)
    pivot_revenue_by_region_category(merged)
    export_and_compare(merged, "cleaned_retail_orders")


if __name__ == "__main__":
    main()
