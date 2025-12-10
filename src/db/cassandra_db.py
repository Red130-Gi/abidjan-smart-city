import logging
import time
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement
from cassandra.policies import DCAwareRoundRobinPolicy

logger = logging.getLogger(__name__)

class CassandraConnector:
    def __init__(self, hosts=['cassandra'], port=9042):
        self.hosts = hosts
        self.port = port
        self.cluster = None
        self.session = None
        self.keyspace = "smart_city"

    def connect(self):
        """Connect to Cassandra cluster."""
        try:
            self.cluster = Cluster(
                self.hosts,
                port=self.port,
                load_balancing_policy=DCAwareRoundRobinPolicy(local_dc='datacenter1'),
                protocol_version=4
            )
            self.session = self.cluster.connect()
            logger.info(f"Connected to Cassandra at {self.hosts}")
            self._initialize_schema()
        except Exception as e:
            logger.error(f"Failed to connect to Cassandra: {e}")
            raise

    def _initialize_schema(self):
        """Initialize keyspace and tables."""
        if not self.session:
            raise Exception("Cassandra session not established")

        # Create Keyspace
        self.session.execute(f"""
            CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
            WITH REPLICATION = {{ 
                'class' : 'SimpleStrategy', 
                'replication_factor' : 1 
            }};
        """)
        
        self.session.set_keyspace(self.keyspace)

        # Create Traffic Data Table
        # Partition Key: segment_id (distribute data by road segment)
        # Clustering Key: timestamp (sort data by time within each segment)
        self.session.execute("""
            CREATE TABLE IF NOT EXISTS traffic_data (
                segment_id text,
                timestamp timestamp,
                vehicle_id text,
                speed float,
                latitude float,
                longitude float,
                vehicle_type text,
                PRIMARY KEY ((segment_id), timestamp)
            ) WITH CLUSTERING ORDER BY (timestamp DESC);
        """)
        
        logger.info("Cassandra schema initialized")

    def close(self):
        if self.cluster:
            self.cluster.shutdown()

# Singleton instance
cassandra_db = CassandraConnector()
