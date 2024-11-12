from psycopg2 import sql
import psycopg2
import boto3
import datetime

def append_data_to_rds_with_iam(data, table_name, db_config, aws_config):
    """
    Appends rows of data to a specified table in a PostgreSQL RDS database using IAM authentication.
    
    Parameters:
    - data (list of tuples): Each tuple represents a row of data to insert.
    - table_name (str): The name of the table in the RDS database.
    - db_config (dict): A dictionary containing database connection details (dbname, host, port).
    - aws_config (dict): Contains AWS authentication details (aws_access_key, aws_secret_key, region, username).
    """
    # Create a session and RDS client
    session = boto3.Session(
        aws_access_key_id=aws_config['aws_access_key'],
        aws_secret_access_key=aws_config['aws_secret_key'],
        region_name=aws_config['region']
    )
    rds_client = session.client('rds')
    
    # Generate an IAM authentication token
    token = rds_client.generate_db_auth_token(
        DBHostname=db_config['host'],
        Port=db_config['port'],
        DBUsername=aws_config['username']
    )
    
    # Connect using the token as the password
    try:
        connection = psycopg2.connect(
            dbname=db_config['dbname'],
            user=aws_config['username'],
            password=token,
            host=db_config['host'],
            port=db_config['port'],
            sslmode='require'
        )
        connection.autocommit = True
        cursor = connection.cursor()
        
        # Prepare SQL insert statement
        insert_query = sql.SQL("INSERT INTO {table} VALUES ({values})").format(
            table=sql.Identifier(table_name),
            values=sql.SQL(', ').join(sql.Placeholder() * len(data[0]))
        )
        
        # Insert each row of data
        for row in data:
            cursor.execute(insert_query, row)
        
        print("Data successfully appended to the table with IAM authentication.")

    except Exception as error:
        print("Error while inserting data:", error)

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
def append_data_to_rds(db_config, table_name, data):
    """
    Append data to a PostgreSQL table on Amazon RDS.
    
    Parameters:
    - db_config: A dictionary with database connection parameters (host, database, user, password, port).
    - table_name: Name of the table where data will be appended.
    - data: List of dictionaries where each dictionary represents a row to be added.
    """
    try:
        # Connect to the PostgreSQL database
        connection = psycopg2.connect(
            host=db_config["host"],
            database=db_config["database"],
            user=db_config["user"],
            password=db_config["password"],
            port=db_config["port"]
        )
        cursor = connection.cursor()
        
        # Identify columns from the first row
        columns = data[0].keys()
        insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(table_name),
            sql.SQL(", ").join(map(sql.Identifier, columns)),
            sql.SQL(", ").join(sql.Placeholder() * len(columns))
        )

        # Execute insert for each row
        for row in data:
            cursor.execute(insert_query, list(row.values()))

        # Commit the transaction
        connection.commit()
        print("Data appended successfully.")

    except Exception as error:
        print("Error while inserting data:", error)
    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        if connection:
            connection.close()


if __name__ == "__main__":
    data = [
    (1, 'John Doe', 29, 'New York'),
    (2, 'Jane Smith', 34, 'San Francisco')
    ]

    table_name = 'your_table_name'
    db_config = {
        'dbname': 'your_database',
        'user': 'your_username',
        'password': 'your_password',
        'host': 'your_rds_host',
        'port': '5432'
    }

    append_data_to_rds(data, table_name, db_config)

