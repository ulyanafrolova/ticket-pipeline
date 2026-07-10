import os

ONELAKE_ENDPOINT = "https://onelake.blob.fabric.microsoft.com"


def load_to_fabric(parquet_path: str, blob_name: str = "Files/processed/tickets_enriched.parquet") -> None:
    """Upload a local file to the Fabric Lakehouse Files section."""
    from azure.identity import ClientSecretCredential
    from azure.storage.blob import BlobServiceClient

    credential = ClientSecretCredential(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        client_id=os.environ["AZURE_CLIENT_ID"],
        client_secret=os.environ["AZURE_CLIENT_SECRET"],
    )
    workspace_id = os.environ["FABRIC_WORKSPACE_ID"]
    lakehouse_id = os.environ["FABRIC_LAKEHOUSE_ID"]

    client = BlobServiceClient(account_url=ONELAKE_ENDPOINT, credential=credential)
    container = f"{workspace_id}/{lakehouse_id}"

    with open(parquet_path, "rb") as f:
        client.get_blob_client(container=container, blob=blob_name).upload_blob(f, overwrite=True)

    print(f"Uploaded {parquet_path} → OneLake {blob_name}")


if __name__ == "__main__":
    load_to_fabric("data/enriched/tickets_enriched.parquet")
