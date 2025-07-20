import lancedb as LB
import pandas as pd
import os
import pyarrow as pa
import numpy as np
from typing import List, Dict, Any
import json

def explore_lancedb_folder(db_path: str, table_name: str = None) -> Dict[str, Any]:
    """
    Explore and display all data in a LanceDB folder
    
    Args:
        db_path: Path to the LanceDB database folder
        table_name: Specific table name to examine (optional)
    
    Returns:
        Dictionary containing database information and data
    """
    print(f"\n{'='*60}")
    print(f"EXPLORING LANCEDB: {db_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(db_path):
        print(f"❌ Database path does not exist: {db_path}")
        return {}
    
    try:
        # Connect to the database
        db = LB.connect(db_path)
        
        # Get all table names
        table_names = db.table_names()
        print(f"📊 Found {len(table_names)} table(s): {table_names}")
        
        db_info = {
            "db_path": db_path,
            "table_count": len(table_names),
            "table_names": table_names,
            "tables_data": {}
        }
        
        # If specific table requested, only process that one
        tables_to_process = [table_name] if table_name and table_name in table_names else table_names
        
        for table_name in tables_to_process:
            print(f"\n🔍 EXAMINING TABLE: {table_name}")
            print("-" * 40)
            
            try:
                table = db.open_table(table_name)
                
                # Get table schema
                schema = table.schema
                print(f"📋 Schema: {schema}")
                
                # Count rows
                try:
                    row_count = table.count_rows()
                    print(f"📊 Total rows: {row_count}")
                except Exception as e:
                    print(f"⚠️  Could not count rows: {e}")
                    row_count = "Unknown"
                
                # Get sample data
                try:
                    # Try to get first 10 rows
                    sample_data = table.head(10).to_pandas()
                    print(f"\n📝 Sample data (first 10 rows):")
                    print(sample_data.to_string(max_rows=10, max_cols=None, width=None))
                    
                    # Show column info
                    print(f"\n📋 Column Information:")
                    for col in sample_data.columns:
                        dtype = sample_data[col].dtype
                        non_null_count = sample_data[col].count()
                        print(f"  - {col}: {dtype} (non-null: {non_null_count})")
                        
                        # Show sample values for text columns
                        if dtype == 'object' and col not in ['embedding']:
                            unique_values = sample_data[col].dropna().unique()[:5]
                            print(f"    Sample values: {list(unique_values)}")
                
                except Exception as e:
                    print(f"⚠️  Could not retrieve sample data: {e}")
                    sample_data = None
                
                # Try to get all data if row count is manageable
                all_data = None
                if isinstance(row_count, int) and row_count <= 1000:
                    try:
                        print(f"\n📄 Retrieving all data ({row_count} rows)...")
                        all_data = table.to_pandas()
                        
                        # Show data statistics
                        print(f"\n📊 Data Statistics:")
                        for col in all_data.columns:
                            if col != 'embedding':  # Skip embedding vectors for readability
                                print(f"  - {col}:")
                                if all_data[col].dtype == 'object':
                                    print(f"    Unique values: {all_data[col].nunique()}")
                                    if all_data[col].nunique() <= 10:
                                        print(f"    Values: {list(all_data[col].unique())}")
                                else:
                                    print(f"    Min: {all_data[col].min()}, Max: {all_data[col].max()}")
                    
                    except Exception as e:
                        print(f"⚠️  Could not retrieve all data: {e}")
                
                # Store table information
                db_info["tables_data"][table_name] = {
                    "schema": str(schema),
                    "row_count": row_count,
                    "sample_data": sample_data.to_dict() if sample_data is not None else None,
                    "all_data": all_data.to_dict() if all_data is not None else None
                }
                
            except Exception as e:
                print(f"❌ Error examining table {table_name}: {e}")
                db_info["tables_data"][table_name] = {"error": str(e)}
        
        return db_info
        
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return {"error": str(e)}

def search_table_content(db_path: str, table_name: str, search_term: str = None, limit: int = 10):
    """
    Search for specific content in a table
    
    Args:
        db_path: Path to the LanceDB database
        table_name: Name of the table to search
        search_term: Term to search for in text fields
        limit: Maximum number of results to return
    """
    print(f"\n🔍 SEARCHING TABLE: {table_name}")
    print(f"Search term: {search_term}")
    print("-" * 40)
    
    try:
        db = LB.connect(db_path)
        table = db.open_table(table_name)
        
        if search_term:
            # Try different search methods
            try:
                # FTS search if available
                results = table.search(search_term).limit(limit).to_pandas()
                print(f"📊 Found {len(results)} results using FTS search:")
                print(results.to_string())
            except:
                try:
                    # Fallback to SQL-like filtering on text columns
                    data = table.to_pandas()
                    text_columns = [col for col in data.columns if data[col].dtype == 'object' and col != 'embedding']
                    
                    mask = pd.Series([False] * len(data))
                    for col in text_columns:
                        mask |= data[col].str.contains(search_term, case=False, na=False)
                    
                    results = data[mask].head(limit)
                    print(f"📊 Found {len(results)} results using text search:")
                    print(results.to_string())
                except Exception as e:
                    print(f"⚠️  Search failed: {e}")
        else:
            # Just show recent entries
            results = table.head(limit).to_pandas()
            print(f"📊 Showing {len(results)} most recent entries:")
            print(results.to_string())
            
    except Exception as e:
        print(f"❌ Error searching table: {e}")

def export_table_data(db_path: str, table_name: str, output_file: str = None):
    """
    Export table data to JSON or CSV file
    
    Args:
        db_path: Path to the LanceDB database
        table_name: Name of the table to export
        output_file: Output file path (optional)
    """
    try:
        db = LB.connect(db_path)
        table = db.open_table(table_name)
        data = table.to_pandas()
        
        if output_file is None:
            output_file = f"{table_name}_export.json"
        
        # Convert embeddings to lists for JSON serialization
        if 'embedding' in data.columns:
            data['embedding'] = data['embedding'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        
        if output_file.endswith('.json'):
            data.to_json(output_file, orient='records', indent=2)
        elif output_file.endswith('.csv'):
            # For CSV, we might want to exclude embeddings due to their size
            csv_data = data.drop('embedding', axis=1) if 'embedding' in data.columns else data
            csv_data.to_csv(output_file, index=False)
        
        print(f"✅ Exported {len(data)} rows to {output_file}")
        
    except Exception as e:
        print(f"❌ Error exporting data: {e}")

def main():
    """
    Main function to explore all LanceDB folders in the workspace
    """
    print("🚀 LANCEDB DATA EXPLORER")
    print("=" * 60)
    
    # Define the database paths from your workspace
    db_paths = [
        ("tmp/user_lancedb", "User Documents Database"),
        ("tmp/skipped_lancedb", "Skipped Documents Database")
    ]
    
    all_db_info = {}
    
    for db_path, description in db_paths:
        print(f"\n🗂️  {description}")
        full_path = os.path.abspath(db_path)
        db_info = explore_lancedb_folder(full_path)
        all_db_info[db_path] = db_info
    
    # Interactive menu
    while True:
        print(f"\n{'='*60}")
        print("INTERACTIVE OPTIONS:")
        print("1. Search in user_docs table")
        print("2. Search in skipped_docs table")
        print("3. Export user_docs to JSON")
        print("4. Export skipped_docs to JSON")
        print("5. Export user_docs to CSV")
        print("6. Export skipped_docs to CSV")
        print("7. Re-examine all databases")
        print("0. Exit")
        print("=" * 60)
        
        choice = input("Enter your choice (0-7): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            search_term = input("Enter search term (or press Enter for recent entries): ").strip()
            search_table_content("tmp/user_lancedb", "user_docs", search_term or None)
        elif choice == "2":
            search_term = input("Enter search term (or press Enter for recent entries): ").strip()
            search_table_content("tmp/skipped_lancedb", "skipped_docs", search_term or None)
        elif choice == "3":
            export_table_data("tmp/user_lancedb", "user_docs", "user_docs_export.json")
        elif choice == "4":
            export_table_data("tmp/skipped_lancedb", "skipped_docs", "skipped_docs_export.json")
        elif choice == "5":
            export_table_data("tmp/user_lancedb", "user_docs", "user_docs_export.csv")
        elif choice == "6":
            export_table_data("tmp/skipped_lancedb", "skipped_docs", "skipped_docs_export.csv")
        elif choice == "7":
            for db_path, description in db_paths:
                print(f"\n🗂️  {description}")
                full_path = os.path.abspath(db_path)
                db_info = explore_lancedb_folder(full_path)
                all_db_info[db_path] = db_info
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()