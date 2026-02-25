"""
Database configuration and connection management for the Career Mapping application.
MongoDB connection disabled - using in-memory fallback.
"""
# from pymongo import MongoClient
# from pymongo.errors import ConnectionFailure
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Database:
    """Mock Database class - MongoDB disabled"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance._initialize_connection()
        return cls._instance
    
    def _initialize_connection(self):
        """Initialize mock connection - MongoDB disabled"""
        logger.warning("MongoDB connection disabled - using in-memory fallback")
        self.db = None
        self.client = None
    
    def get_collection(self, collection_name):
        """Get a collection by name - MongoDB disabled"""
        logger.warning(f"MongoDB disabled - cannot access collection: {collection_name}")
        return None
    
    def close_connection(self):
        """Close the MongoDB connection - MongoDB disabled"""
        logger.info("MongoDB connection already disabled")

# Create a single instance of the database
db = Database()

def get_db():
    """Get the database instance - MongoDB disabled"""
    return db

def init_db():
    """Initialize database - MongoDB disabled"""
    logger.warning("Database initialization skipped - MongoDB disabled")
    return True
            
            # Test the connection
            self.client.admin.command('ping')
            
            # Initialize database
            self.db = self.client[self.DB_NAME]
            
            # Initialize collections with validation (optional)
            self._initialize_collections()
            
            logger.info(f"Successfully connected to MongoDB at {self.MONGODB_URI}")
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def _initialize_collections(self):
        """Initialize collections with validation and indexes"""
        # Users collection
        if 'users' not in self.db.list_collection_names():
            self.db.create_collection('users')
        self.users = self.db.users
        
        # Assessments collection
        if 'assessments' not in self.db.list_collection_names():
            self.db.create_collection('assessments')
        self.assessments = self.db.assessments
        
        # Conversations collection
        if 'conversations' not in self.db.list_collection_names():
            self.db.create_collection('conversations')
        self.conversations = self.db.conversations
        
        # Create indexes
        self._create_indexes()
    
    def _create_indexes(self):
        """Create necessary indexes for better query performance"""
        # Users collection indexes
        self.users.create_index('email', unique=True, sparse=True)
        self.users.create_index('username', unique=True, sparse=True)
        
        # Assessments collection indexes
        self.assessments.create_index([('user_id', 1), ('timestamp', -1)])
        self.assessments.create_index('stream')
        
        # Conversations collection indexes
        self.conversations.create_index([('user_id', 1), ('updated_at', -1)])
        
        logger.info("Database indexes created/verified")
    
    def get_collection(self, collection_name):
        """Get a collection by name"""
        if hasattr(self, collection_name):
            return getattr(self, collection_name)
        return self.db[collection_name]
    
    def close_connection(self):
        """Close the MongoDB connection"""
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("MongoDB connection closed")

# Create a single instance of the database
db = Database()

def get_db():
    """Get the database instance"""
    return db

def init_db():
    """Initialize database with indexes and initial data if needed"""
    try:
        db = get_db()
        logger.info("Database initialization complete")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

# Initialize database when this module is imported
if __name__ != '__main__':
    init_db()

# Example usage:
if __name__ == '__main__':
    # Test database connection
    try:
        db = get_db()
        print("Successfully connected to MongoDB!")
        print(f"Available collections: {db.db.list_collection_names()}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close_connection()
