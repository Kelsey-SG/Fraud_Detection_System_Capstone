HYBRID FRAUD ANOMALY DETECTION IN MOBILE MONEY TRANSACTIONS
Using a Long-Range Transformer Model and Relational Database Anomaly Scoring
================================================================================

--------------------------------------------------------------------------------
1. PROJECT OVERVIEW
--------------------------------------------------------------------------------

This project builds a fraud detection system for mobile money
transactions. There are two components to the system:

  1. A Long-range Transformer model that takes behavioural feature data from
     mobile money transactions. It outputs a reconstruction error as well
     as a score for the likelihood of fraud.

  2. A PostgreSQL database based anomaly scorer that applies a set of
     rules to detect anomalies in the database (such as balance mismatch,
     zero-drain transactions, large amounts, high velocity transactions,
     and jump to destination transactions).

These two scores are combined (with a 0.8 / 0.2 default weighting in
favour of the Transformer model) at deployment time. The threshold for
considering transactions as “fraud” is chosen to maximise the F1 score
for the fraud class. The system was developed and evaluated using the
PaySim dataset.

--------------------------------------------------------------------------------
2. SOURCE CODE REPOSITORY
--------------------------------------------------------------------------------

GitHub repository: https://github.com/Kelsey-SG/Fraud_Detection_System_Capstone.git


--------------------------------------------------------------------------------
3. SOURCE FILES (/code)
--------------------------------------------------------------------------------

  model_components.ipynb / model_components.py
      Defines the architecture of the Long-range Transformer model, as well as
      the methods for engineering the behavioural features and creating the
      sliding window that will feed data to the model. This file is imported
      within both the train_model and deploy_model notebooks.

  train_model.ipynb
      This notebook contains the steps necessary to train the
      Long-Range Transformer model. It loads the PaySim dataset, engineering
      the features, splitting the data into training (70%), validation (15%),
      and test (15%) partitions, down-sampling the normal transactions to a
      ratio of 1:10 relative to fraud transactions, training the model,
      and saving the trained model and its components to the /resources
      directory.

  db_scoring.ipynb / db_scoring.py
      This Python file contains the interface between the Transformer model
      and the relational database. Methods are defined for connecting to the
      database, inserting data, scoring individual transactions, batching
      scores, real-time scoring, flagging transactions as potentially
      fraudulent, and reading views of the database. This file also includes
      a standalone demo of the database methods.

  deploy_model.ipynb
      This notebook defines the steps required to deploy the model. After
      loading the trained model, the model is deployed in a way that fuses
      the Transformer and database scores, selects a threshold for fraud
      classification, and writes the results and flagged transactions to the
      database.

--------------------------------------------------------------------------------
4. SETUP AND DEPENDENCIES
--------------------------------------------------------------------------------

4.1 System requirements
  - Python 3.9 or newer
  - 8 GB RAM minimum (16 GB recommended for full PaySim training)
  - GPU optional but speeds up training significantly
  - PostgreSQL database (the project was developed against Neon serverless)

4.2 Python packages
  Install with pip (versions known to work; newer versions are usually fine):

    pip install pandas numpy scikit-learn scipy
    pip install torch transformers
    pip install psycopg2-binary
    pip install jupyter notebook

  Alternatively, run the first cell of any notebook — each one re-installs
  the packages it needs at the top.

4.3 Database setup
  The notebooks expect a PostgreSQL database with the following objects:

    Tables:
      transactions
      anomaly_scores
      flagged_transactions

    Functions:
      compute_sql_anomaly_score(uuid)
      score_transaction_batch(uuid[])

    Views:
      v_rule_score_summary
      v_rule_hit_rates
      v_high_risk_transactions

  Connection details are loaded from environment variables (see 5.4).
  The SQL DDL for these objects is included in /resources/sql_schema.sql
  if available; otherwise see Appendix B of the report.

4.4 Environment variables
  Database credentials are NEVER hard-coded. Set the following before
  running db_scoring.ipynb or deploy_model.ipynb:

    export DB_HOST="your.postgres.host"
    export DB_PORT="5432"
    export DB_NAME="your_database"
    export DB_USER="your_user"
    export DB_PASSWORD="your_password"
    export DB_SSLMODE="require"

  And, optionally, the project root if it is not your current directory:

    export CAPSTONE_BASE_DIR="/path/to/project/root"

--------------------------------------------------------------------------------
5. HOW TO RUN
--------------------------------------------------------------------------------

5.1 Get the data
  Download the PaySim dataset (e.g. from Kaggle) and place it at:

    $CAPSTONE_BASE_DIR/data/paysim.csv

5.2 Train the model
  Open and run train_model.ipynb top to bottom. By default it uses 20
  epochs and a context length of 20. The notebook saves:

    trained_model/model_weights.pt
    trained_model/scaler.pkl
    trained_model/label_encoder.pkl
    trained_model/config.json
    data/test_data.csv

5.3 Stand up the database (optional, but recommended)
  The SQL schema necessary to create the anomaly database is located
  within the sql_schema.sql fil. Run this script on your PostgreSQL database. 
  Export the DB\_\* environment variables above. The system can be run without 
  the database, but the database anomaly scores will be disregarded.

5.4 Run the standalone DB demo (optional)
  In db_scoring.ipynb, run the Standalone Demo cell. It connects to the
  database, inserts a small slice of data, scores it, and prints a summary.

5.5 Deploy
  Open and run deploy_model.ipynb. It loads the saved model, runs
  inference on data/test_data.csv, fuses the result with SQL scores
  (or zeros if the database is unavailable), and writes
  deployment_results.csv to the project root.
