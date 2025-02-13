import textwrap
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import json
from huggingface_hub import InferenceClient
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore")

# Define the dataset analysis class
class DatasetAnalysisTool:
    def __init__(self):
        """Initialize with no dataset loaded initially."""
        self.data = None
        self.client = InferenceClient(api_key="WriteYourAPIKeyHere")
        self.encoders = {}  # Initialize the dictionary to store label encoded columns
        self.one_hot_columns = []  # Initialize the list to store one-hot encoded columns
        self.generated_plots = []  # Initialize the list to store generated plots

    def load_data(self):
        """Load dataset based on user input for the file path."""
        filepath = input("Enter the file path of the dataset: ")
        try:
            self.data = pd.read_csv(filepath)
            print("Dataset loaded successfully!")
        except Exception as e:
            print(f"Error loading dataset: {e}")

    def explore_data(self):
        """Basic exploration of the dataset."""
        if self.data is not None:
            print("\n--- Dataset Information ---\n")
            print(self.data.info())
            print("\n--- First 5 Rows ---\n")
            print(self.data.head())
            print("\n--- Summary Statistics ---\n")
            print(self.data.describe())

    def clean_data(self):
        """Handle missing values and outliers interactively."""
        if self.data is not None:
            print("\nHandling missing values...")
            # Drop rows with missing values in string columns
            string_cols = self.data.select_dtypes(include=['object']).columns
            if string_cols.any():
                self.data.dropna(subset=string_cols, inplace=True)
                print(f"Rows with missing values in string columns dropped.")

            # Fill missing values in numeric columns with column medians
            numeric_cols = self.data.select_dtypes(include=np.number).columns
            if numeric_cols.any():
                self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].median())
                print("Missing values in numeric columns filled with column medians.")


    def analyze_data(self):
        """Perform advanced analysis with AI assistance to determine relationships and visualize them."""
        if self.data is not None:
            print("\n--- Analyzing Relationships in the Dataset with AI Assistance ---\n")

            # Step 1: Create a summary of the dataset
            summary = {
                "columns": list(self.data.columns),
                "shape": self.data.shape,
                "numeric_summary": self.data.describe().to_dict(),
                "sample_data": self.data.sample(n=10, random_state=42).to_dict(orient="records")  #"sample_data": self.data.head(5).to_dict(orient="records"), 
            }

            # Step 2: Format the summary into a prompt for the AI model
            prompt = (
                "You are an advanced AI model specialized in data analysis and visualization. Your task "
                "is to analyze a dataset and determine relationships between each pair of columns. You must "
                "identify the most interesting relationships based on the column names, data types, and statistical properties. "
                "For each interesting relationship, recommend the best visualization method and"
                "take into account that some columns may contain a large number of elements, so"
                "**recommend the best visualization method STRICTLY from the following list:**\n\n"
                "- scatterplot\n"
                "- boxplot\n"
                "- heatmap\n"
                "- pairplot\n"
                "- barplot\n"
                "- histogram\n"
                "- violinplot\n"
                "- lineplot\n"
                "- 3d scatterplot\n"
                "- stacked bar chart\n"
                "- countplot\n"
                "- time series plot\n"
                "- lag plot\n"
                "- area plot\n"
                "- pie chart\n"
                "- bubble chart\n"
                "- density plot\n"
                "- step plot\n\n"
                "### **Instructions:**\n"
                "1. **Identify Relationships:** Analyze every pair of columns in the dataset and categorize their relationship type "
                "(e.g., correlation, categorical association, time series trend, distribution comparison, etc.).\n"
                "2. **Determine Interesting Relationships:** Prioritize the most insightful relationships based on the column names, "
                "data types, statistical dependencies, and potential for meaningful insights.\n"
                "3. **Select the Best Visualization:** Recommend the most effective visualization from the provided list based on the "
                "relationship type and data characteristics while taking into account that some columns may contain a large number of elements, so recommend the best visualization method accordingly."
                "recommend at least as many plots as the number of columns divided by 2.\n"
                "4. **Format the Response in JSON:** The response must strictly follow this JSON format:\n\n"
                "{\n"
                '  "relationships": [\n'
                '    {\n'
                '      "columns": ["column_1", "column_2"],\n'
                '      "relationship_type": "type of relationship (e.g., correlation, categorical association, trend, distribution)",\n'
                '      "recommended_visualization": "visualization type from the provided list"\n'
                '    },\n'
                '    ...\n'
                '  ]\n'
                "}\n\n"
                "### **Dataset Summary:**\n"
                f"Columns: {summary['columns']}\n"
                f"Shape: {summary['shape']}\n"
                f"Numeric Summary: {summary['numeric_summary']}\n"
                f"Sample Data: {summary['sample_data']}\n"
                "Ensure the response is valid JSON and does not contain any additional text or explanations."
            )

            # Step 3: Query the AI model
            try:
                response = self.client.chat.completions.create(
                    model="mistralai/Mistral-7B-Instruct-v0.3",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2500
                )
                ai_response = response["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"Error querying the AI model: {e}")
                return

            # Validate the AI response
            if not ai_response.strip():
                print("AI response is empty or invalid.")
                return

            print("\n--- AI Analysis and Recommendations ---\n")
            print(ai_response)

            # Step 4: Parse AI response for relationships and visualizations
            try:
                relationships = self._parse_ai_response(ai_response)
                print(f"Parsed relationships: {relationships}")
            except Exception as e:
                print(f"Error parsing AI response: {e}")
                return

            # Validate parsed relationships
            if not relationships:
                print("No valid relationships found for visualization.")
                return

            # Step 5: Generate visualizations for each recommended relationship
            for relationship in relationships:
                try:
                    columns = relationship["columns"]
                    visualization = relationship.get("recommended_visualization", "").lower()
                    if not visualization:
                        print(f"Skipping visualization for {columns}: No visualization type provided.")
                        continue
                    print(f"\nGenerating {visualization} for {columns}...")
                    self._generate_visualization(columns, visualization)
                except Exception as e:
                    print(f"Error generating visualization for {columns}: {e}")

    def _parse_ai_response(self, ai_response):
        """Parse the AI's response, which is expected to be in JSON format."""
        relationships = []
        try:
            # Debug: Print raw AI response
            print("\n--- Raw AI Response ---\n")
            print(ai_response)

            # Parse the AI response as JSON
            parsed_response = json.loads(ai_response)

            # Extract relationships
            relationships = parsed_response.get("relationships", [])
            print("\n--- Parsed Relationships ---\n")
            for relationship in relationships:
                print(f"Columns: {relationship['columns']}, Visualization: {relationship['recommended_visualization']}")
        except json.JSONDecodeError as e:
            print(f"Error decoding AI response as JSON: {e}")
        except KeyError as e:
            print(f"Missing key in AI response: {e}")
        except Exception as e:
            print(f"Unexpected error parsing AI response: {e}")
        
        return relationships

    def _generate_visualization(self, columns, visualization):
        """Generate the appropriate visualizations for the given columns."""
        try:
            # Split visualization types if multiple are provided (e.g. "bar plot or pie chart")
            visualization_options = [v.strip() for v in visualization.split(" or ")]

            # Single visualization case
            if len(visualization_options) == 1:
                fig, ax = plt.subplots(figsize=(8, 6))
                self._plot_individual_visualization(ax, columns, visualization_options[0])
                plt.tight_layout()
                plt.show()

            # Multiple visualizations (side-by-side)
            elif len(visualization_options) == 2:
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                self._plot_individual_visualization(axes[0], columns, visualization_options[0])
                self._plot_individual_visualization(axes[1], columns, visualization_options[1])
                plt.tight_layout()
                plt.show()

            else:
                print(f"Unsupported number of visualizations for: {visualization_options}")

            # Save or display the plot
            headless = matplotlib.get_backend() == "agg"
            filename = f"visualization_{'_'.join(columns)}.png"
            if headless:
                fig.savefig(filename)
                print(f"Plot saved as {filename}")
            else:
                plt.show()

        except Exception as e:
            print(f"Error generating visualization for {columns}: {e}")


    def _plot_individual_visualization(self, ax, columns, visualization):
        """Plot a single visualization on a given axis."""
        try:
            if visualization == "scatterplot" or visualization == "scatter plot":
                sns.scatterplot(x=self.data[columns[0]], y=self.data[columns[1]], ax=ax)
                ax.set_title(f"Scatterplot: {columns[0]} vs {columns[1]}")
                ax.set_xlabel(columns[0])
                ax.set_ylabel(columns[1])

            elif visualization == "heatmap" or visualization == "heat map":
                contingency_table = pd.crosstab(self.data[columns[0]], self.data[columns[1]])
                sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
                ax.set_title(f"Heatmap: {columns[0]} vs {columns[1]}")
                ax.set_ylabel(columns[0])
                ax.set_xlabel(columns[1])

            elif visualization == "boxplot" or visualization == "box plot":
                sns.boxplot(x=self.data[columns[0]], y=self.data[columns[1]], ax=ax)
                ax.set_title(f"Boxplot: {columns[1]} by {columns[0]}")
                ax.set_xlabel(columns[0])
                ax.set_ylabel(columns[1])

            elif visualization == "pairplot" or visualization == "pair plot":
                sns.pairplot(self.data[columns])  # Pairplot cannot use `ax`
                plt.suptitle(f"Pairplot: {', '.join(columns)}", y=1.02)
                plt.show()

            elif visualization == "barplot" or visualization == "bar plot":
                sns.barplot(x=self.data[columns[0]], y=self.data[columns[1]], ax=ax)
                ax.set_title(f"Barplot: {columns[0]} vs {columns[1]}")
                ax.set_xlabel(columns[0])
                ax.set_ylabel(columns[1])

            elif visualization == "histogram" or visualization == "hist":
                sns.histplot(data=self.data, x=columns[0], bins=30, kde=True, ax=ax)
                ax.set_title(f"Histogram of {columns[0]}")
                ax.set_xlabel(columns[0])
                ax.set_ylabel("Frequency")

            elif visualization == "violinplot" or visualization == "violin plot":
                sns.violinplot(x=self.data[columns[0]], y=self.data[columns[1]], ax=ax)
                ax.set_title(f"Violin Plot: {columns[1]} by {columns[0]}")
                ax.set_xlabel(columns[0])
                ax.set_ylabel(columns[1])

            elif visualization == "lineplot" or visualization == "line plot":
                sns.lineplot(x=self.data[columns[0]], y=self.data[columns[1]], ax=ax)
                ax.set_title(f"Line Plot: {columns[0]} vs {columns[1]}")
                ax.set_xlabel(columns[0])
                ax.set_ylabel(columns[1])

            elif visualization == "3d scatterplot" or visualization == "3d scatter plot":
                from mpl_toolkits.mplot3d import Axes3D
                ax = plt.figure().add_subplot(111, projection='3d')
                ax.scatter(self.data[columns[0]], self.data[columns[1]], self.data[columns[2]])
                ax.set_title(f"3D Scatterplot: {columns[0]}, {columns[1]}, {columns[2]}")
                ax.set_xlabel(columns[0])
                ax.set_ylabel(columns[1])
                ax.set_zlabel(columns[2])
                plt.show()

            elif visualization == "stacked bar chart" or visualization == "stacked bar":
                contingency_table = pd.crosstab(self.data[columns[0]], self.data[columns[1]])
                contingency_table.plot(kind="bar", stacked=True, ax=ax)
                ax.set_title(f"Stacked Bar Chart: {columns[0]} vs {columns[1]}")
                ax.set_xlabel(columns[0])
                ax.set_ylabel("Frequency")

            elif visualization == "countplot" or visualization == "count plot":
                sns.countplot(x=self.data[columns[0]], ax=ax)
                ax.set_title(f"Count Plot: {columns[0]}")
                ax.set_xlabel(columns[0])
                ax.set_ylabel("Count")

            elif visualization == "time series plot" or visualization == "time plot":
                sns.lineplot(x=self.data[columns[0]], y=self.data[columns[1]], ax=ax)
                ax.set_title(f"Time Series Plot: {columns[0]} vs {columns[1]}")
                ax.set_xlabel(columns[0])
                ax.set_ylabel(columns[1])

            elif visualization == "lag plot":
                pd.plotting.lag_plot(self.data[columns[0]], ax=ax)
                ax.set_title(f"Lag Plot: {columns[0]}")

            elif visualization == "area plot":
                self.data.plot.area(x=columns[0], y=columns[1], ax=ax)
                ax.set_title(f"Area Plot: {columns[0]} vs {columns[1]}")
                ax.set_xlabel(columns[0])
                ax.set_ylabel(columns[1])

            elif visualization == "pie chart":
                self.data[columns[0]].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
                ax.set_title(f"Pie Chart for {columns[0]}")
                ax.set_ylabel("Proportion")

            elif visualization == "bubble chart":
                sns.scatterplot(x=self.data[columns[0]], y=self.data[columns[1]], size=self.data[columns[2]], ax=ax)
                ax.set_title(f"Bubble Chart: {columns[0]} vs {columns[1]}")
                ax.set_xlabel(columns[0])
                ax.set_ylabel(columns[1])

            elif visualization == "density plot":
                sns.kdeplot(x=self.data[columns[0]], ax=ax)
                ax.set_title(f"Density Plot of {columns[0]}")
                ax.set_xlabel(columns[0])
                ax.set_ylabel("Density")

            elif visualization == "step plot":
                ax.step(self.data[columns[0]], self.data[columns[1]], where="mid")
                ax.set_title(f"Step Plot: {columns[0]} vs {columns[1]}")
                ax.set_xlabel(columns[0])
                ax.set_ylabel(columns[1])

            else:
                ax.text(0.5, 0.5, f"'{visualization}' not recognized", ha="center", va="center")
                ax.set_title("Unknown Visualization")

                # Rotate x-axis labels if they are too cluttered
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')

                # Rotate y-axis labels if they are too cluttered
            for label in ax.get_yticklabels():
                label.set_rotation(45)
                label.set_ha('right')

            # Adjust font size
            ax.tick_params(axis='both', which='major', labelsize=10)

            # Adjust layout to prevent overlap
            plt.tight_layout()

        except Exception as e:
            print(f"Error plotting {visualization} for {columns}: {e}")

    def determine_target_variable(self):
        """Try to identify the target variable for the dataset interactively."""
        if self.data is not None:
            print("\n--- Target Variable Selection ---")
            print("Based on the columns and summary, please choose a target variable for the model.")
            print("Available columns:")
            print(self.data.columns)
            self.target_variable = input("Enter the name of the target variable: ")
            if self.target_variable in self.data.columns:
                print(f"Target variable set to: {self.target_variable}")
            else:
                print(f"Error: {self.target_variable} not found in dataset.")

    def encode_categorical_columns(self):
        """Automatically detect and encode categorical columns."""
        if self.data is not None:
            # Detect categorical columns
            categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
            if not categorical_cols:
                print("No categorical columns detected.")
                return

            print("\n--- Categorical Column Encoding ---")
            print(f"Detected categorical columns: {categorical_cols}")
            
            # Ask user how they want to encode each categorical column
            for col in categorical_cols:
                encoding_method = input(f"How would you like to encode '{col}'? (1) Label Encoding (2) One-Hot Encoding: ")
                
                if encoding_method == "1":
                    # Label encoding
                    le = LabelEncoder()
                    self.data[col] = le.fit_transform(self.data[col])
                    self.encoders[col] = le  # Store the encoder
                    print(f"Label Encoding applied to {col}")
                elif encoding_method == "2":
                    # One-hot encoding
                    self.data = pd.get_dummies(self.data, columns=[col], drop_first=True)
                    self.one_hot_columns.append(col)  # Store the column name
                    print(f"One-Hot Encoding applied to {col}")
                else:
                    print(f"Invalid choice. Skipping encoding for {col}.")
    
    def train_models(self):
        """Train a model and baseline model on the dataset with detailed evaluation and cross-validation."""
        if self.data is not None and self.target_variable is not None:
            X = self.data.drop(columns=[self.target_variable])
            y = self.data[self.target_variable]

            # Split the dataset into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Determine if the task is classification or regression
            if y.nunique() <= 20:  # Classification (small number of unique target values)
                self.model = TPOTClassifier(verbosity=2, generations=3, population_size=20, random_state=42)
                # Cross-validation (2-fold)
                self.cross_val_scores = cross_val_score(self.model, X, y, cv=2, scoring="accuracy")
                model_type = 'classification'
                baseline_model_type = DecisionTreeClassifier(random_state=42)
            else:  # Regression (large number of unique target values)
                self.model = TPOTRegressor(verbosity=2, generations=3, population_size=20, random_state=42)
                # Cross-validation (2-fold)
                self.cross_val_scores = cross_val_score(self.model, X, y, cv=2, scoring="neg_root_mean_squared_error")
                model_type = 'regression'
                baseline_model_type = DecisionTreeRegressor(random_state=42)

            # Training the selected model
            self.model.fit(X_train, y_train)

            # Training the baseline model
            self.baseline_model = baseline_model_type
            self.baseline_model.fit(X_train, y_train)

            # Model evaluation
            y_pred = self.model.predict(X_test)
            baseline_pred = self.baseline_model.predict(X_test)

            if model_type == 'classification':  # Classification task
                model_accuracy = accuracy_score(y_test, y_pred)
                baseline_accuracy = accuracy_score(y_test, baseline_pred)
                print(f"\nModel accuracy: {model_accuracy:.4f}")
                print(f"Baseline model accuracy: {baseline_accuracy:.4f}")
                print("\nClassification Report (Precision, Recall, F1-Score):")
                print(classification_report(y_test, y_pred))

                self.model_test_results = {
                    'model_accuracy': model_accuracy,
                    'baseline_accuracy': baseline_accuracy,
                    'classification_report': classification_report(y_test, y_pred)
                }
            else:  # Regression task
                model_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
                print(f"\nModel RMSE: {model_rmse:.4f}")
                print(f"Baseline model RMSE: {baseline_rmse:.4f}")
                
                model_r2 = r2_score(y_test, y_pred)
                baseline_r2 = r2_score(y_test, baseline_pred)
                print(f"\nModel R²: {model_r2:.4f}")
                print(f"Baseline model R²: {baseline_r2:.4f}")

                self.model_test_results = {
                    'model_rmse': model_rmse,
                    'baseline_rmse': baseline_rmse,
                    'model_r2': model_r2,
                    'baseline_r2': baseline_r2
                }

            # Log models and evaluation in MLflow
            mlflow.start_run()
            mlflow.log_param("target_variable", self.target_variable)
            mlflow.log_metric("model_accuracy" if model_type == 'classification' else "model_rmse", model_accuracy if model_type == 'classification' else model_rmse)
            mlflow.log_metric("baseline_accuracy" if model_type == 'classification' else "baseline_rmse", baseline_accuracy if model_type == 'classification' else baseline_rmse)
            mlflow.sklearn.log_model(self.model, "trained_model")
            mlflow.sklearn.log_model(self.baseline_model, "baseline_model")
            mlflow.end_run()

            print("\n--- Models and evaluation logged to MLflow ---")

    def generate_report(self):
        """Generate a detailed analysis report including graphs, target variable, and model results."""
        if self.data is not None:
            try:
                # Reverse Label Encoding for categorical columns
                decoded_data = self.data.copy()
                for col, encoder in self.encoders.items():
                    decoded_data[col] = encoder.inverse_transform(decoded_data[col])

                # Reverse One-Hot Encoding for columns (if applicable)
                for col in self.one_hot_columns:
                    original_col_name = col.split("_")[0]
                    decoded_data[original_col_name] = decoded_data.filter(regex=f"^{col}_").idxmax(axis=1).str.replace(f"^{col}_", "")

                # Prepare a summary of the dataset to send to the model
                summary = {
                    "columns": list(self.data.columns),
                    "shape": self.data.shape,
                    "numeric_summary": self.data.describe().to_dict(),
                    "correlation_analysis": "A heatmap of correlations among numeric columns was generated.",
                    "distribution_analysis": "Distributions of numeric columns were plotted."
                }

                # Include the determined target variable if available
                if hasattr(self, 'target_variable'):
                    summary["target_variable"] = self.target_variable
                else:
                    summary["target_variable"] = "Target variable not determined."

                # Include model training and evaluation results if available
                model_results = {}
                if hasattr(self, 'model'):
                    model_results["cross_validation_scores"] = getattr(self, "cross_val_scores", "Not available")
                    model_results["test_results"] = getattr(self, "model_test_results", "Not available")
                else:
                    model_results["model_training"] = "No model has been trained yet."

                summary["model_results"] = model_results

                # Prepare the report prompt with the generated plots
                prompt = (
                    "You are an advanced data analyst. Based on the following dataset analysis, "
                    "including generated graphs, the target variable, and model results, "
                    "write a detailed report summarizing the findings. The report should have the following sections with the respective word counts:\n\n"
                    "1. **Introduction**: Around 150 words describing the purpose of the dataset analysis and the context.\n"
                    "2. **Data Quality Checks**: Max 150 words discussing the cleaning, handling of missing values, and outliers in the dataset.\n"
                    "3. **Data Exploration**: Between 250 to 500 words analyzing the dataset, summarizing key insights, correlation findings, and distribution analysis.\n"
                    "4. **Training a Model**: Around 300 words explaining how the model was trained, the algorithms used, and cross-validation steps.\n"
                    "5. **Model Evaluation**: Around 300 words discussing the evaluation results, comparing model performance, and including key metrics like accuracy, RMSE, R², etc.\n"
                    "6. **Conclusion**: Around 300 words summarizing the findings regarding the dataset and model performance along with any recommendations or insights.\n\n"
                    f"Dataset shape: {summary['shape']}\n"
                    f"Columns: {summary['columns']}\n"
                    f"Summary statistics: {summary['numeric_summary']}\n"
                    f"Correlation analysis: {summary['correlation_analysis']}\n"
                    f"Distribution analysis: {summary['distribution_analysis']}\n"
                    f"Target variable: {summary['target_variable']}\n"
                    f"Model results: {summary['model_results']}\n"
                )

                # Send the prompt to Mistral with plots included
                if self.generated_plots:
                    for fig in self.generated_plots:
                        fig_filename = "plot.png"
                        fig.savefig(fig_filename)
                        with open(fig_filename, "rb") as f:
                            plot_data = f.read()

                messages = [
                    {"role": "user", "content": prompt}
                ]

                completion = self.client.chat.completions.create(
                    model="mistralai/Mistral-7B-Instruct-v0.3",
                    messages=messages, 
                    max_tokens=2500
                )

                # Save the model's response to a text file
                report_content = completion.choices[0].message['content']

                # Word wrapping: Wrap each paragraph to fit within 80 characters
                wrapper = textwrap.TextWrapper(width=80, expand_tabs=False, replace_whitespace=False)
                wrapped_report = wrapper.fill(report_content)

                with open("dataset_analysis_report.txt", "w") as report_file:
                    report_file.write("=== Dataset Analysis Report ===\n\n")
                    report_file.write("Generated Report:\n")
                    report_file.write("---------------------------------------------\n\n")

                    paragraphs = wrapped_report.split("\n\n")
                    for paragraph in paragraphs:
                        if paragraph.strip():
                            report_file.write(f"{paragraph.strip()}\n\n")

                print("\n--- Report saved to 'dataset_analysis_report.txt' ---\n")

            except Exception as e:
                print(f"Error generating report: {e}")

# Interactive execution
if __name__ == "__main__":
    analyzer = DatasetAnalysisTool()

    while True:
        print("\n--- Dataset Analysis Menu ---")
        print("1. Load Dataset")
        print("2. Explore Dataset")
        print("3. Clean Dataset")
        print("4. Determine Target Variable")
        print("5. Analyze Dataset")
        print("6. Encode Categorical Columns")
        print("7. Train Models and Evaluate")
        print("8. Generate Report")
        print("9. Exit")

        choice = input("Enter your choice (1-9): ")

        if choice == "1":
            analyzer.load_data()
        elif choice == "2":
            analyzer.explore_data()
        elif choice == "3":
            analyzer.clean_data()
        elif choice == "4":
            analyzer.determine_target_variable()
        elif choice == "5":
            analyzer.analyze_data()
        elif choice == "6":
            analyzer.encode_categorical_columns()
        elif choice == "7":
            analyzer.train_models()
        elif choice == "8":
            analyzer.generate_report()
        elif choice == "9":
            print("Exiting the application. Goodbye!")
            break
        else:
            print("Invalid choice. Please select a valid option.")