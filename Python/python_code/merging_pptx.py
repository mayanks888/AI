# Import module
import groupdocs_merger_cloud
from shutil import copyfile

# Get your client Id and Secret at https://dashboard.groupdocs.cloud (free registration is required).
clientid = "xxxxx-xxxx-xxxx-xxxx-xxxxxxxx"
clientsecret = "xxxxxxxxxxxxxxxxxxxxxxxxx"

# Create instance of the API
documentApi = groupdocs_merger_cloud.DocumentApi.from_keys(clientid, clientsecret)
file_api = groupdocs_merger_cloud.FileApi.from_keys(clientid, clientsecret)

try:

    #upload source files to default storage
    filename1 = 'C:/Temp/test.pptx'
    remote_name1 = 'slides/test.pptx'
    filename2 = 'C:/Temp/three-slides.pptx'
    remote_name2 = 'slides/three-slides.pptx'

    output_name= 'slides/joined.pptx'


    request_upload1 = groupdocs_merger_cloud.UploadFileRequest(remote_name1,filename1)
    response_upload1 = file_api.upload_file(request_upload1)
    request_upload2 = groupdocs_merger_cloud.UploadFileRequest(remote_name2,filename2)
    response_upload2 = file_api.upload_file(request_upload2)

    item1 = groupdocs_merger_cloud.JoinItem()
    item1.file_info = groupdocs_merger_cloud.FileInfo(remote_name1)
    item2 = groupdocs_merger_cloud.JoinItem()
    item2.file_info = groupdocs_merger_cloud.FileInfo(remote_name2)

    options = groupdocs_merger_cloud.JoinOptions()
    options.join_items = [item1, item2]
    options.output_path = output_name

    result = documentApi.join(groupdocs_merger_cloud.JoinRequest(options))


    #Download Document from default Storage
    request_download = groupdocs_merger_cloud.DownloadFileRequest(output_name)
    response_download = file_api.download_file(request_download)

    copyfile(response_download, 'C:/Temp/joined.pptx')
    print("Result {}".format(response_download))

except groupdocs_merger_cloud.ApiException as e:
        print("Exception when converting document: {0}".format(e.message))
