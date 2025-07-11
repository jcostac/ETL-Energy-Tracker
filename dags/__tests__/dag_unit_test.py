import unittest
from airflow.models import DagBag

class TestESIOSPreciosDag(unittest.TestCase):
    def setUp(self):
        self.dagbag = DagBag()
        
    def test_dag_loaded(self):
        dag = self.dagbag.get_dag("esios_precios_etl")
        self.assertIsNotNone(dag)
        self.assertEqual(len(dag.tasks), 6)  # Check if all 6 tasks are present
    
    def test_dependencies(self):
        dag = self.dagbag.get_dag("esios_precios_etl")
        
        # Test the task dependencies
        extract_task = dag.get_task("extract_esios_prices")
        check_extract_task = dag.get_task("check_extraction")
        transform_task = dag.get_task("transform_esios_prices")
        check_transform_task = dag.get_task("check_transform")
        load_task = dag.get_task("load_esios_prices_to_datalake")
        finalize_task = dag.get_task("finalize_pipeline")
        
        # Check upstream/downstream relationships
        self.assertIn(check_extract_task, extract_task.downstream_list)
        self.assertIn(transform_task, check_extract_task.downstream_list)
        self.assertIn(check_transform_task, transform_task.downstream_list)
        self.assertIn(load_task, check_transform_task.downstream_list)
        self.assertIn(finalize_task, load_task.downstream_list)

    def test_no_import_errors(self):
     self.assertEqual(len(self.dagbag.import_errors), 0, f"DAG import errors: {self.dagbag.import_errors}")



