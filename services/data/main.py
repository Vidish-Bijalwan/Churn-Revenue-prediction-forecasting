"""
ForeTel.AI Enterprise Data Service
Advanced Data Pipeline with Real-time Processing and Quality Monitoring
"""

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Integer, Text, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, validator
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import redis
import kafka
from kafka import KafkaProducer, KafkaConsumer
import asyncio
import logging
import json
from great_expectations.core import ExpectationSuite
from great_expectations.dataset import PandasDataset
import boto3
from cryptography.fernet import Fernet
import hashlib
from io import StringIO
import pyarrow as pa
import pyarrow.parquet as pq
from feast import FeatureStore
import dask.dataframe as dd
from dask.distributed import Client

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

# Initialize FastAPI
app = FastAPI(
    title="ForeTel.AI Data Service",
    description="Enterprise data pipeline with real-time processing and quality monitoring",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Initialize services
redis_client = redis.from_url(REDIS_URL)
kafka_producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
    value_serializer=lambda x: json.dumps(x, default=str).encode('utf-8')
)

# AWS S3 client
s3_client = boto3.client('s3')

# Encryption
cipher_suite = Fernet(ENCRYPTION_KEY.encode())

# Dask client for distributed processing
dask_client = Client('scheduler-address:8786')  # Configure based on setup

# Database Models
class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(Text)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer)
    rows = Column(Integer)
    columns = Column(Integer)
    schema = Column(JSON)
    quality_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    tags = Column(JSON)
    metadata = Column(JSON)

class DataQualityCheck(Base):
    __tablename__ = "data_quality_checks"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, nullable=False)
    check_type = Column(String, nullable=False)
    check_name = Column(String, nullable=False)
    expected_value = Column(String)
    actual_value = Column(String)
    status = Column(String, nullable=False)  # PASS, FAIL, WARNING
    timestamp = Column(DateTime, default=datetime.utcnow)
    details = Column(JSON)

class DataLineage(Base):
    __tablename__ = "data_lineage"
    
    id = Column(Integer, primary_key=True, index=True)
    source_dataset_id = Column(Integer)
    target_dataset_id = Column(Integer)
    transformation = Column(String, nullable=False)
    transformation_code = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic Models
class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    tags: Optional[List[str]] = []

class DatasetResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    file_path: str
    file_size: int
    rows: int
    columns: int
    quality_score: float
    created_at: datetime
    tags: List[str]

class DataQualityRequest(BaseModel):
    dataset_id: int
    checks: List[Dict[str, Any]]

class FeatureEngineeringRequest(BaseModel):
    dataset_id: int
    transformations: List[Dict[str, Any]]
    target_name: str

class StreamingDataRequest(BaseModel):
    topic: str
    data: Dict[str, Any]
    timestamp: Optional[datetime] = None

# Advanced Data Processing Classes
class DataQualityEngine:
    def __init__(self):
        self.expectation_suites = {}
    
    def create_expectation_suite(self, dataset_name: str) -> ExpectationSuite:
        """Create data quality expectation suite"""
        suite = ExpectationSuite(expectation_suite_name=f"{dataset_name}_quality_suite")
        
        # Add common expectations
        suite.expectations.extend([
            {"expectation_type": "expect_table_row_count_to_be_between", "kwargs": {"min_value": 1000}},
            {"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {"column": "customer_id"}},
            {"expectation_type": "expect_column_values_to_be_unique", "kwargs": {"column": "customer_id"}},
            {"expectation_type": "expect_column_values_to_be_in_type_list", "kwargs": {"column": "age", "type_list": ["int", "float"]}},
            {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "age", "min_value": 18, "max_value": 100}},
        ])
        
        self.expectation_suites[dataset_name] = suite
        return suite
    
    def validate_dataset(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Validate dataset against expectations"""
        if dataset_name not in self.expectation_suites:
            self.create_expectation_suite(dataset_name)
        
        # Convert to Great Expectations dataset
        ge_df = PandasDataset(df)
        
        # Run validation
        results = []
        for expectation in self.expectation_suites[dataset_name].expectations:
            try:
                result = getattr(ge_df, expectation["expectation_type"])(**expectation["kwargs"])
                results.append({
                    "expectation": expectation["expectation_type"],
                    "success": result.success,
                    "result": result.result
                })
            except Exception as e:
                results.append({
                    "expectation": expectation["expectation_type"],
                    "success": False,
                    "error": str(e)
                })
        
        # Calculate quality score
        success_rate = sum(1 for r in results if r["success"]) / len(results)
        
        return {
            "quality_score": success_rate,
            "checks": results,
            "summary": {
                "total_checks": len(results),
                "passed": sum(1 for r in results if r["success"]),
                "failed": sum(1 for r in results if not r["success"])
            }
        }

class FeatureStore:
    def __init__(self):
        self.features = {}
        self.feature_groups = {}
    
    def register_feature_group(self, name: str, features: List[str], description: str = ""):
        """Register a feature group"""
        self.feature_groups[name] = {
            "features": features,
            "description": description,
            "created_at": datetime.utcnow()
        }
    
    def compute_features(self, df: pd.DataFrame, feature_group: str) -> pd.DataFrame:
        """Compute features for a feature group"""
        if feature_group == "customer_demographics":
            return self._compute_demographic_features(df)
        elif feature_group == "usage_patterns":
            return self._compute_usage_features(df)
        elif feature_group == "financial_metrics":
            return self._compute_financial_features(df)
        else:
            return df
    
    def _compute_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute demographic features"""
        df_features = df.copy()
        
        # Age groups
        df_features['age_group'] = pd.cut(df_features['age'], 
                                        bins=[0, 25, 35, 50, 65, 100], 
                                        labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        
        # Income brackets
        df_features['income_bracket'] = pd.cut(df_features.get('monthly_income', 0), 
                                             bins=[0, 25000, 50000, 75000, 100000, float('inf')], 
                                             labels=['Low', 'Medium-Low', 'Medium', 'High', 'Very High'])
        
        return df_features
    
    def _compute_usage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute usage pattern features"""
        df_features = df.copy()
        
        # Usage intensity
        if 'data_usage_gb' in df_features.columns:
            df_features['usage_intensity'] = pd.cut(df_features['data_usage_gb'], 
                                                  bins=[0, 5, 15, 30, float('inf')], 
                                                  labels=['Light', 'Medium', 'Heavy', 'Very Heavy'])
        
        # Call patterns
        if 'call_usage_min' in df_features.columns:
            df_features['call_pattern'] = pd.cut(df_features['call_usage_min'], 
                                               bins=[0, 100, 300, 600, float('inf')], 
                                               labels=['Minimal', 'Low', 'Medium', 'High'])
        
        return df_features
    
    def _compute_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute financial features"""
        df_features = df.copy()
        
        # Revenue per customer
        if 'total_revenue' in df_features.columns and 'months_active' in df_features.columns:
            df_features['revenue_per_month'] = df_features['total_revenue'] / df_features['months_active']
        
        # Spending ratio
        if 'monthly_spending' in df_features.columns and 'monthly_income' in df_features.columns:
            df_features['spending_ratio'] = df_features['monthly_spending'] / df_features['monthly_income']
        
        return df_features

class StreamingProcessor:
    def __init__(self):
        self.consumers = {}
        self.processors = {}
    
    async def start_consumer(self, topic: str, processor_func):
        """Start Kafka consumer for real-time processing"""
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        self.consumers[topic] = consumer
        
        # Process messages
        for message in consumer:
            try:
                processed_data = await processor_func(message.value)
                
                # Send processed data to output topic
                kafka_producer.send(f"{topic}_processed", processed_data)
                
                # Cache in Redis for real-time access
                redis_client.setex(
                    f"streaming:{topic}:{datetime.now().timestamp()}", 
                    3600, 
                    json.dumps(processed_data, default=str)
                )
                
            except Exception as e:
                logging.error(f"Error processing streaming data: {e}")
    
    async def process_customer_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process real-time customer data"""
        # Add timestamp
        data['processed_at'] = datetime.utcnow().isoformat()
        
        # Compute real-time features
        if 'data_usage_gb' in data:
            data['usage_category'] = 'high' if data['data_usage_gb'] > 20 else 'normal'
        
        # Detect anomalies
        data['anomaly_score'] = self._calculate_anomaly_score(data)
        
        return data
    
    def _calculate_anomaly_score(self, data: Dict[str, Any]) -> float:
        """Calculate anomaly score for real-time data"""
        # Simplified anomaly detection
        score = 0.0
        
        if data.get('data_usage_gb', 0) > 50:
            score += 0.3
        if data.get('call_usage_min', 0) > 1000:
            score += 0.2
        if data.get('monthly_spending', 0) > 200:
            score += 0.2
        
        return min(score, 1.0)

# Initialize engines
quality_engine = DataQualityEngine()
feature_store = FeatureStore()
streaming_processor = StreamingProcessor()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Utility functions
def encrypt_sensitive_data(data: str) -> str:
    return cipher_suite.encrypt(data.encode()).decode()

def decrypt_sensitive_data(encrypted_data: str) -> str:
    return cipher_suite.decrypt(encrypted_data.encode()).decode()

def upload_to_s3(file_content: bytes, file_name: str) -> str:
    """Upload file to S3 and return URL"""
    try:
        s3_client.put_object(Bucket=S3_BUCKET, Key=file_name, Body=file_content)
        return f"s3://{S3_BUCKET}/{file_name}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {str(e)}")

# Routes
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "data",
        "timestamp": datetime.utcnow(),
        "redis_connected": redis_client.ping(),
        "dask_connected": dask_client.status == "running"
    }

@app.post("/datasets/upload", response_model=DatasetResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_info: DatasetCreate = Depends(),
    db: Session = Depends(get_db)
):
    """Upload and process dataset"""
    try:
        # Read file content
        content = await file.read()
        
        # Parse CSV
        df = pd.read_csv(StringIO(content.decode('utf-8')))
        
        # Generate file path
        file_name = f"datasets/{dataset_info.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Upload to S3
        s3_path = upload_to_s3(content, file_name)
        
        # Run data quality checks
        quality_results = quality_engine.validate_dataset(df, dataset_info.name)
        
        # Create dataset record
        dataset = Dataset(
            name=dataset_info.name,
            description=dataset_info.description,
            file_path=s3_path,
            file_size=len(content),
            rows=len(df),
            columns=len(df.columns),
            schema=df.dtypes.to_dict(),
            quality_score=quality_results["quality_score"],
            tags=dataset_info.tags,
            metadata={
                "original_filename": file.filename,
                "upload_timestamp": datetime.utcnow().isoformat(),
                "quality_checks": quality_results["checks"]
            }
        )
        
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        
        # Store quality check results
        for check in quality_results["checks"]:
            quality_check = DataQualityCheck(
                dataset_id=dataset.id,
                check_type="expectation",
                check_name=check["expectation"],
                status="PASS" if check["success"] else "FAIL",
                details=check
            )
            db.add(quality_check)
        
        db.commit()
        
        # Send notification
        kafka_producer.send("dataset_uploaded", {
            "dataset_id": dataset.id,
            "name": dataset.name,
            "quality_score": quality_results["quality_score"],
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            file_path=dataset.file_path,
            file_size=dataset.file_size,
            rows=dataset.rows,
            columns=dataset.columns,
            quality_score=dataset.quality_score,
            created_at=dataset.created_at,
            tags=dataset.tags or []
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset upload failed: {str(e)}")

@app.get("/datasets", response_model=List[DatasetResponse])
async def list_datasets(db: Session = Depends(get_db)):
    """List all datasets"""
    datasets = db.query(Dataset).filter(Dataset.is_active == True).all()
    
    return [
        DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            file_path=dataset.file_path,
            file_size=dataset.file_size,
            rows=dataset.rows,
            columns=dataset.columns,
            quality_score=dataset.quality_score,
            created_at=dataset.created_at,
            tags=dataset.tags or []
        )
        for dataset in datasets
    ]

@app.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Get dataset details"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Get quality checks
    quality_checks = db.query(DataQualityCheck).filter(
        DataQualityCheck.dataset_id == dataset_id
    ).all()
    
    return {
        "dataset": dataset,
        "quality_checks": quality_checks,
        "schema": dataset.schema,
        "metadata": dataset.metadata
    }

@app.post("/datasets/{dataset_id}/quality-check")
async def run_quality_check(
    dataset_id: int,
    request: DataQualityRequest,
    db: Session = Depends(get_db)
):
    """Run data quality checks on dataset"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        # Load dataset from S3
        # This would load the actual data from S3
        # For now, we'll simulate the process
        
        # Run custom quality checks
        results = []
        for check in request.checks:
            # Simulate check execution
            result = {
                "check_name": check["name"],
                "check_type": check["type"],
                "status": "PASS",  # Would be actual result
                "details": check
            }
            results.append(result)
            
            # Store in database
            quality_check = DataQualityCheck(
                dataset_id=dataset_id,
                check_type=check["type"],
                check_name=check["name"],
                status=result["status"],
                details=result["details"]
            )
            db.add(quality_check)
        
        db.commit()
        
        return {"results": results, "summary": {"total": len(results)}}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality check failed: {str(e)}")

@app.post("/datasets/{dataset_id}/features")
async def engineer_features(
    dataset_id: int,
    request: FeatureEngineeringRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Engineer features for dataset"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Start feature engineering in background
    background_tasks.add_task(
        _engineer_features_background,
        dataset_id,
        request.transformations,
        request.target_name
    )
    
    return {"status": "feature_engineering_started", "dataset_id": dataset_id}

async def _engineer_features_background(dataset_id: int, transformations: List[Dict], target_name: str):
    """Background task for feature engineering"""
    try:
        # Load dataset (simulate)
        # df = load_from_s3(dataset.file_path)
        
        # Apply transformations
        for transformation in transformations:
            if transformation["type"] == "demographic_features":
                # df = feature_store.compute_features(df, "customer_demographics")
                pass
            elif transformation["type"] == "usage_features":
                # df = feature_store.compute_features(df, "usage_patterns")
                pass
            elif transformation["type"] == "financial_features":
                # df = feature_store.compute_features(df, "financial_metrics")
                pass
        
        # Save engineered dataset
        # new_file_path = upload_to_s3(df.to_csv().encode(), f"features/{target_name}.csv")
        
        # Send completion notification
        kafka_producer.send("feature_engineering_completed", {
            "dataset_id": dataset_id,
            "target_name": target_name,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        kafka_producer.send("feature_engineering_failed", {
            "dataset_id": dataset_id,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })

@app.post("/streaming/ingest")
async def ingest_streaming_data(request: StreamingDataRequest):
    """Ingest real-time streaming data"""
    try:
        # Process data
        processed_data = await streaming_processor.process_customer_data(request.data)
        
        # Send to Kafka
        kafka_producer.send(request.topic, processed_data)
        
        # Cache in Redis
        cache_key = f"streaming:{request.topic}:{datetime.now().timestamp()}"
        redis_client.setex(cache_key, 3600, json.dumps(processed_data, default=str))
        
        return {
            "status": "ingested",
            "processed_data": processed_data,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming ingestion failed: {str(e)}")

@app.get("/streaming/{topic}/latest")
async def get_latest_streaming_data(topic: str, limit: int = 100):
    """Get latest streaming data for topic"""
    try:
        # Get from Redis
        keys = redis_client.keys(f"streaming:{topic}:*")
        keys.sort(reverse=True)  # Latest first
        
        data = []
        for key in keys[:limit]:
            cached_data = redis_client.get(key)
            if cached_data:
                data.append(json.loads(cached_data))
        
        return {"topic": topic, "data": data, "count": len(data)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get streaming data: {str(e)}")

@app.get("/lineage/{dataset_id}")
async def get_data_lineage(dataset_id: int, db: Session = Depends(get_db)):
    """Get data lineage for dataset"""
    lineage = db.query(DataLineage).filter(
        (DataLineage.source_dataset_id == dataset_id) | 
        (DataLineage.target_dataset_id == dataset_id)
    ).all()
    
    return {"dataset_id": dataset_id, "lineage": lineage}

@app.post("/datasets/{dataset_id}/backup")
async def backup_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Backup dataset to multiple locations"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        # Create backup in different S3 bucket/region
        backup_path = f"backups/{dataset.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Copy to backup location (simulate)
        # s3_client.copy_object(...)
        
        return {
            "status": "backed_up",
            "backup_path": backup_path,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backup failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
