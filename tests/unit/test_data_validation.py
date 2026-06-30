import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from mlops_utils.data_validation import DataValidator, DataCheckResult

@pytest.fixture(scope="module")
def spark():
    return SparkSession.builder.master("local[1]").appName("test_data_validation").getOrCreate()

@pytest.fixture
def dummy_df(spark):
    schema = StructType([
        StructField("id", StringType(), True),
        StructField("category", StringType(), True),
        StructField("value", DoubleType(), True),
        StructField("age", IntegerType(), True),
    ])
    data = [
        ("1", "A", 10.5, 25),
        ("2", "B", 0.0, 30),
        ("3", "A", -5.0, 15),
        ("4", "C", 100.0, 45),
        (None, "B", None, 120),
    ]
    return spark.createDataFrame(data, schema)

def test_data_validator_all_pass(dummy_df):
    df = dummy_df.filter(dummy_df.id.isNotNull()) # Drop the null id row for success tests
    
    validator = DataValidator(df)
    validator.add_check(*DataValidator.check_no_nulls("id"))
    validator.add_check(*DataValidator.check_unique("id"))
    validator.add_check(*DataValidator.check_allowed_values("category", ["A", "B", "C"]))
    validator.add_check(*DataValidator.check_range("age", 0, 150))
    validator.add_check(*DataValidator.check_custom_sql("value_not_too_low", "value >= -10.0"))
    
    passed, results = validator.run(raise_on_fail=True)
    assert passed is True
    assert len(results) == 5
    assert all(r.passed for r in results)

def test_data_validator_no_nulls_fails(dummy_df):
    validator = DataValidator(dummy_df)
    validator.add_check(*DataValidator.check_no_nulls("id"))
    
    passed, results = validator.run(raise_on_fail=False)
    assert passed is False
    assert results[0].passed is False
    assert "Found 1 nulls in 'id'" in results[0].message
    
    with pytest.raises(ValueError, match="Data Quality Validation failed"):
        validator.run(raise_on_fail=True)

def test_data_validator_unique_fails(spark):
    data = [("1",), ("1",), ("2",)]
    df = spark.createDataFrame(data, ["id"])
    
    validator = DataValidator(df)
    validator.add_check(*DataValidator.check_unique("id"))
    
    passed, results = validator.run(raise_on_fail=False)
    assert passed is False
    assert "Total: 3, Distinct: 2" in results[0].message

def test_data_validator_allowed_values_fails(dummy_df):
    validator = DataValidator(dummy_df)
    validator.add_check(*DataValidator.check_allowed_values("category", ["A", "B"])) # Missing 'C'
    
    passed, results = validator.run(raise_on_fail=False)
    assert passed is False
    assert "Found 1 invalid values for 'category'" in results[0].message

def test_data_validator_range_fails(dummy_df):
    validator = DataValidator(dummy_df)
    validator.add_check(*DataValidator.check_range("age", 20, 50)) # 15 and 120 are out of bounds
    
    passed, results = validator.run(raise_on_fail=False)
    assert passed is False
    assert "Found 2 values out of bounds [20, 50] in 'age'" in results[0].message

def test_data_validator_custom_sql_fails(dummy_df):
    validator = DataValidator(dummy_df)
    validator.add_check(*DataValidator.check_custom_sql("value_positive", "value > 0")) # 0.0, -5.0, None are failing
    
    passed, results = validator.run(raise_on_fail=False)
    assert passed is False
    assert "Found 3 rows failing expression: value > 0" in results[0].message
