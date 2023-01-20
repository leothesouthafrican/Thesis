from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes


#initiate the authentication client
credential = DefaultAzureCredential()



ml_client = MLClient(
    credential = credential,
    subscription_id = "6dce6c27-66cd-4856-92e5-fc861b474681",
    resource_group_name = "thesis_compute",
    workspace_name = "Thesis"
)

my_path = '/Users/leo/Desktop/Thesis/data/faces_2000'

my_data = Data(
    path=my_path,
    type=AssetTypes.URI_FOLDER,
    description="MSFT generated data 2000 faces",
    name="faces2000",
    version='1'
)

ml_client.data.create_or_update(my_data)