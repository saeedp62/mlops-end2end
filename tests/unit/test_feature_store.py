import pytest
from unittest.mock import MagicMock, patch

from mlops_utils.feature_store import (
    FeatureStoreManager,
    create_or_replace_feature_table,
    write_feature_table,
    build_training_set,
    score_batch_wrapper,
    create_feature_serving_endpoint,
    publish_online_if_enabled,
)

@pytest.fixture
def mock_fe_client():
    return MagicMock()

@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.catalog = "test_catalog"
    cfg.schemas.offline_features = "offline_schema"
    cfg.schemas.online_features = "online_schema"
    cfg.feature_table = "test_table"
    cfg.primary_keys = ["id"]
    cfg.online_store.enabled = True
    return cfg

@pytest.fixture
def fsm(mock_fe_client, mock_config):
    return FeatureStoreManager(
        fe=mock_fe_client,
        catalog=mock_config.catalog,
        offline_schema=mock_config.schemas.offline_features,
        online_schema=mock_config.schemas.online_features,
        dry_run=False
    )

class TestFeatureStoreManager:
    def test_from_config(self, mock_config):
        with patch('databricks.feature_engineering.FeatureEngineeringClient') as mock_client:
            fsm = FeatureStoreManager.from_config(mock_config)
            assert fsm.catalog == "test_catalog"
            assert fsm.offline_schema == "offline_schema"
            assert fsm.online_schema == "online_schema"
            assert fsm.dry_run is False

    def test_fqn_offline(self, fsm):
        assert fsm.fqn("my_table") == "test_catalog.offline_schema.my_table"

    def test_fqn_online(self, fsm):
        assert fsm.fqn("my_table", use_online_schema=True) == "test_catalog.online_schema.my_table"

    def test_create_or_replace(self, fsm, mock_fe_client):
        df_mock = MagicMock()
        fsm.create_or_replace("my_table", df_mock, ["id"], description="test")
        mock_fe_client.create_table.assert_called_once()
        kwargs = mock_fe_client.create_table.call_args[1]
        assert kwargs["name"] == "test_catalog.offline_schema.my_table"
        assert kwargs["primary_keys"] == ["id"]
        assert kwargs["description"] == "test"

    def test_write(self, fsm, mock_fe_client):
        df_mock = MagicMock()
        fsm.write("my_table", df_mock, mode="merge")
        mock_fe_client.write_table.assert_called_once_with(
            name="test_catalog.offline_schema.my_table",
            df=df_mock,
            mode="merge"
        )

    def test_reset_and_write(self, fsm):
        with patch.object(fsm, 'create_or_replace') as mock_create, patch.object(fsm, 'write') as mock_write:
            df_mock = MagicMock()
            fsm.reset_and_write("my_table", df_mock, ["id"])
            mock_create.assert_called_once_with("my_table", df_mock, ["id"])
            mock_write.assert_called_once_with("my_table", df_mock)

    def test_build_training_set(self, fsm, mock_fe_client):
        df_mock = MagicMock()
        mock_ts = MagicMock()
        mock_fe_client.create_training_set.return_value = mock_ts
        
        res = fsm.build_training_set(df_mock, [], "label")
        mock_fe_client.create_training_set.assert_called_once()
        mock_ts.load_df.assert_called_once()
        assert res == mock_ts.load_df.return_value

    def test_score_batch(self, fsm, mock_fe_client):
        df_mock = MagicMock()
        fsm.score_batch(df_mock, "model_uri")
        mock_fe_client.score_batch.assert_called_once_with(
            df=df_mock,
            model_uri="model_uri",
            result_type="string",
            env_manager="virtualenv"
        )

    def test_create_lookups(self, fsm):
        with patch('databricks.feature_engineering.FeatureLookup') as mock_lookup:
            lookups = fsm.create_lookups("my_table", ["id"])
            assert len(lookups) == 1
            mock_lookup.assert_called_once_with(
                table_name="test_catalog.offline_schema.my_table",
                lookup_key=["id"],
                feature_names=None,
                rename_features=None,
                timestamp_lookup_key=None
            )

    def test_drop_table(self, fsm, mock_fe_client):
        fsm.drop_table("my_table")
        mock_fe_client.drop_table.assert_called_once_with(name="test_catalog.offline_schema.my_table")

    def test_read_table(self, fsm, mock_fe_client):
        fsm.read_table("my_table")
        mock_fe_client.read_table.assert_called_once_with(name="test_catalog.offline_schema.my_table")

    def test_set_tags(self, fsm, mock_fe_client):
        tags = {"k1": "v1", "k2": "v2"}
        fsm.set_tags("my_table", tags)
        assert mock_fe_client.set_feature_table_tag.call_count == 2

    def test_update_description(self, fsm):
        with patch('mlops_utils.spark_utils.get_or_create_spark') as mock_spark:
            fsm.update_description("my_table", "new 'desc'")
            mock_spark.return_value.sql.assert_called_once_with(
                "COMMENT ON TABLE test_catalog.offline_schema.my_table IS 'new \\'desc\\''"
            )

    def test_sync_online_table_not_exists(self, fsm):
        with patch('databricks.sdk.WorkspaceClient') as mock_wc:
            mock_w = mock_wc.return_value
            mock_w.online_tables.get.side_effect = Exception("Not found")
            
            fsm.sync_online_table("my_table", ["id"])
            
            mock_w.online_tables.get.assert_called_once_with("test_catalog.online_schema.my_table")
            mock_w.online_tables.create.assert_called_once()
            args, kwargs = mock_w.online_tables.create.call_args
            assert kwargs["name"] == "test_catalog.online_schema.my_table"
            assert kwargs["spec"].source_table_full_name == "test_catalog.offline_schema.my_table"
            assert kwargs["spec"].primary_key_columns == ["id"]

    def test_sync_online_table_exists(self, fsm):
        with patch('databricks.sdk.WorkspaceClient') as mock_wc:
            mock_w = mock_wc.return_value
            # .get does not raise
            
            fsm.sync_online_table("my_table", ["id"])
            mock_w.online_tables.get.assert_called_once()
            mock_w.online_tables.create.assert_not_called()

    def test_query_endpoint(self, fsm):
        with patch('databricks.sdk.WorkspaceClient') as mock_wc:
            mock_w = mock_wc.return_value
            records = [{"id": 1}]
            mock_response = MagicMock()
            mock_response.predictions = [0.5]
            mock_w.serving_endpoints.query.return_value = mock_response
            
            res = fsm.query_endpoint("my_endpoint", records)
            mock_w.serving_endpoints.query.assert_called_once_with(
                name="my_endpoint",
                dataframe_records=records
            )
            assert res == [0.5]

    def test_dry_run_bypasses_mutations(self, fsm, mock_fe_client):
        fsm.dry_run = True
        
        # Test methods that should be bypassed
        fsm.drop_table("my_table")
        fsm.set_tags("my_table", {"k": "v"})
        fsm.update_description("my_table", "test")
        fsm.sync_online_table("my_table", ["id"])
        
        # fe shouldn't be touched for these
        mock_fe_client.drop_table.assert_not_called()
        mock_fe_client.set_feature_table_tag.assert_not_called()
        
        with patch('mlops_utils.spark_utils.get_or_create_spark') as mock_spark:
            mock_spark.assert_not_called()
            
        with patch('databricks.sdk.WorkspaceClient') as mock_wc:
            mock_wc.assert_not_called()


def test_publish_online_if_enabled(fsm, mock_config):
    with patch.object(fsm, 'sync_online_table') as mock_sync:
        res = publish_online_if_enabled(mock_config, fsm)
        assert res is True
        mock_sync.assert_called_once_with(
            table_name=mock_config.feature_table,
            primary_keys=mock_config.primary_keys
        )

def test_publish_online_if_disabled(fsm, mock_config):
    mock_config.online_store.enabled = False
    with patch.object(fsm, 'sync_online_table') as mock_sync:
        res = publish_online_if_enabled(mock_config, fsm)
        assert res is False
        mock_sync.assert_not_called()

def test_create_feature_serving_endpoint_creates():
    mock_w = MagicMock()
    with patch('databricks.sdk.service.serving.EndpointCoreConfigInput', create=True):
        res = create_feature_serving_endpoint("my_endpoint", [], workspace_client=mock_w)
        mock_w.serving_endpoints.create_and_wait.assert_called_once()
        mock_w.serving_endpoints.update_config_and_wait.assert_not_called()
        assert res == mock_w.serving_endpoints.create_and_wait.return_value

def test_create_feature_serving_endpoint_updates_on_exception():
    mock_w = MagicMock()
    mock_w.serving_endpoints.create_and_wait.side_effect = Exception("Already exists")
    
    with patch('databricks.sdk.service.serving.EndpointCoreConfigInput', create=True):
        res = create_feature_serving_endpoint("my_endpoint", [], workspace_client=mock_w)
        mock_w.serving_endpoints.create_and_wait.assert_called_once()
        mock_w.serving_endpoints.update_config_and_wait.assert_called_once()
        assert res == mock_w.serving_endpoints.update_config_and_wait.return_value
