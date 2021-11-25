import os
import json
import requests
import warnings
from requests_toolbelt import MultipartEncoder

class MultiDimDocmentApi(object):
    """
    一次只能针对一个飞书一个多维表格的一个数据表进行操作
    """
    def __init__(self, user_id, app_id, app_secret, app_token, table_id) -> None:
        """

        Args:
            user_id (str): 用户主体的ID，在管理后台->组织架构->成员->点击自己的头像
            app_id (str): 创建应用的app id
            app_secret (str): 创建应用的app secret
            app_token (str): 需要操作的多维表格的token
            table_id (str): 需要操作的数据表token
        """
        super().__init__()
        self.user_id = user_id
        self.app_id = app_id
        self.app_secret = app_secret
        self.app_token = app_token
        self.table_id = table_id
        pass

    def get_authorization_token(self):
        """由app id得到认证token

        Returns:
            str: token
        """
        # 获取凭证
        url= "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal/"   
        post_data = {"app_id": self.app_id, "app_secret": self.app_secret}
        responese = requests.post(url, data=post_data)
        authorization_token = responese.json()["tenant_access_token"] 
        return authorization_token

    def add_record(self, record_dict):
        """向表格添加一行记录，该函数必须保证需要添加的columns都已被创建，否则将失败

        Args:
            record_dict (dict): 新增记录的信息

        Returns:
            Response: 响应
        """
        authorization_token = self.get_authorization_token()
        
        # 新增记录请求地址
        url = "https://open.feishu.cn/open-apis/bitable/v1/apps/{}/tables/{}/records".format(self.app_token, self.table_id)
        # 请求Header（字典形式储存）
        header = {"Content-Type":"application/json; charset=utf-8", "Authorization":"Bearer " + str(authorization_token)}
        # 发送POST请求
        response = requests.post(url, data=json.dumps(record_dict), headers=header)
        return response

    def list_columns(self):
        """返回所有字段的信息列表

        Returns:
            list: 所有字段的信息列表
        """
        authorization_token = self.get_authorization_token()
        url = "https://open.feishu.cn/open-apis/bitable/v1/apps/{}/tables/{}/fields".format(self.app_token, self.table_id)
        header = {"Authorization":"Bearer " + str(authorization_token)}
        response = requests.get(url, headers = header)
        return response.json()["data"]["items"]

    def add_column(self, column_info_dict):
        """添加一列

        Args:
            column_info (dict): 添加列的info dict，参照 https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/bitable-v1/app-table-field/create 中的请求体部分

        Returns:
            Response: 返回响应
        """
        authorization_token = self.get_authorization_token()
        url = "https://open.feishu.cn/open-apis/bitable/v1/apps/{}/tables/{}/fields".format(self.app_token, self.table_id)
        header = {"Content-Type":"application/json; charset=utf-8", "Authorization":"Bearer " + str(authorization_token)}
        response = requests.post(url, data=json.dumps(column_info_dict).encode("utf-8"), headers = header)
        return response

    def update_column(self, field_id, new_field_info_dict):
        """更新table的特定field的属性，可用于添加多选框的选项

        Args:
            field_id (str): 需要修改的field的id
            new_field_info_dict (dict): 包括属性修改的列表

        Returns:
            Response: 返回响应
        """
        authorization_token = self.get_authorization_token()
        url = "https://open.feishu.cn/open-apis/bitable/v1/apps/{}/tables/{}/fields/{}".format(self.app_token, self.table_id, field_id)
        header = {"Content-Type":"application/json; charset=utf-8", "Authorization":"Bearer " + str(authorization_token)}
        response = requests.put(url, data=json.dumps(new_field_info_dict).encode("utf-8"), headers=header)
        return response

    def upload_file(self, file_path):
        """上传素材到当前多维表格

        Args:
            file_path (str): 要上传的文件路径

        Returns:
            Response: 返回响应
            str: 上传完成的素材的token
        """
        authorization_token = self.get_authorization_token()

        url = "https://open.feishu.cn/open-apis/drive/v1/medias/upload_all"

        f = open(file_path,'rb')
        data = MultipartEncoder(
            fields={'file_name': file_path, "parent_type": "bitable_file", "parent_node": self.app_token, "size": str(os.stat(file_path).st_size), "file": f}
        )

        headers = {
            'Authorization': 'Bearer {}'.format(authorization_token),
            'Content-Type': data.content_type
        }

        response = requests.post(url, headers=headers, data=data)
        file_token = json.loads(response.text)["data"]["file_token"]

        f.close()
        return response, file_token

    def get_root_meta(self):
        """获取”我的空间元信息“

        Returns:
            str: folder_id
        """
        authorization_token = self.get_authorization_token()
        url = "https://open.feishu.cn/open-apis/drive/explorer/v2/root_folder/meta"
        headers = {'Authorization': 'Bearer {}'.format(authorization_token)}
        response = requests.get(url, headers=headers)
        return json.loads(response.text)["data"]["token"]

    def change_file_permission(self, file_token, user_id, perm="edit"):
        """改变特定文件的特定用户权限，阅读or编辑

        Args:
            file_token (str): 
            user_id (str): 
            perm (str, optional): read or edit. Defaults to "edit".

        Returns:
            Response: 返回响应
        """
        authorization_token = self.get_authorization_token()
        url = "https://open.feishu.cn/open-apis/drive/permission/member/update"

        headers = {
            'Authorization': 'Bearer ' + authorization_token,
            'Content-Type': 'application/json; charset=utf-8'
        }

        payload = {
            "token": file_token,
            "type": "bitable",
            "member_type": "userid",
            "member_id": user_id,
            "perm": perm,
            "notify_lark": "true"
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        return response

    # def smart_new_bitable(self, title):  # 目前弃用，用不上了
    #     """利用API创建一个所有者为应有的多维表格

    #     Args:
    #         title ([type]): [description]

    #     Returns:
    #         [type]: [description]
    #     """
    #     authorization_token = self.get_authorization_token()
        
    #     # 获取当前用户主体得云空间根目录文件夹token
    #     root_folder_token = self.get_root_meta()

    #     # 新建多维表格
    #     url = "https://open.feishu.cn/open-apis/drive/explorer/v2/file/{}".format(root_folder_token)
    #     headers = {
    #         'Authorization': 'Bearer {}'.format(authorization_token),
    #         'Content-Type': 'application/json; charset=utf-8'
    #     }
    #     payload = {"type": "bitable", "title": title}
    #     response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    #     file_token = json.loads(response.text)["data"]["token"]

    #     # 为用户主体添加权限
    #     response = self.change_file_permission(file_token, user_id=self.user_id, perm="edit")

    #     return response

    def smart_add_record(self, record_dict):
        """智能添加记录，自动检查有效性、新增列

        Args:
            record_dict (dict): 新增记录的dict

        Returns:
            Response: 返回响应
        """
        # 得到table中已有字段信息
        exists_columns_list = self.list_columns()
        exists_columns_names = [i["field_name"] for i in exists_columns_list]
        exists_columns_types = [i["type"] for i in exists_columns_list]
        
        error_columns_names = []
        final_record_dict = {}
        # 遍历需要新增record的field，检查需要处理的field
        for ii, (key, value) in enumerate(record_dict.items()):
            # 检查是否存在，若不存在则新增一个多行文本类型
            if key not in exists_columns_names:
                if isinstance(value, list):
                    value = [str(i) for i in value]
                    new_column_info_dict = {
                        "field_name": key,
                        "type": 4,
                        "property": {
                            "options": [{"name": i} for i in value]
                        }
                    }
                    self.add_column(new_column_info_dict)
                    final_record_dict[key] = value
                else: 
                    new_column_info_dict = {
                        "field_name": key,
                        "type": 1
                    }
                    self.add_column(new_column_info_dict)
                    final_record_dict[key] = str(value)
            else:
                # 如果存在，判断表格中对应field的类别并判定新增value的有效性，若无效则放入错误fileds中后续删除
                index = exists_columns_names.index(key)
                exist_column_type = exists_columns_types[index]
                if exist_column_type == 2:
                    try:
                        value = float(value)
                        final_record_dict[key] = value
                    except BaseException:
                        error_columns_names.append(key)
                elif exist_column_type == 4:  # 多选
                    if not isinstance(value, list):
                        value = [value]
                    value = [str(i) for i in value]
                    options = exists_columns_list[index]["property"]["options"]
                    options_names = [i["name"] for i in options]
                    options = [{"name": i} for i in set(options_names + value)]
                    new_field_info_dict = {
                        "property": {
                            "options": options
                        }
                    }
                    self.update_column(exists_columns_list[index]["field_id"], new_field_info_dict)
                    final_record_dict[key] = value
                elif exist_column_type == 3:
                    value = str(value)
                    options = exists_columns_list[index]["property"]["options"]
                    options_names = [i["name"] for i in options]
                    options = [{"name": i} for i in set(options_names + [value])]
                    new_field_info_dict = {
                        "property": {
                            "options": options
                        }
                    }
                    self.update_column(exists_columns_list[index]["field_id"], new_field_info_dict)
                    final_record_dict[key] = value
                elif exist_column_type == 17:
                    assert isinstance(value, str) or isinstance(value, list)
                    if isinstance(value, str): value = [value]
                    file_tokens = []
                    for file_path in value:
                        if not os.path.exists(file_path):
                            warnings.warn("文件路径：{} 不存在，跳过该文件上传".format(file_path))
                            continue
                        _, file_token = self.upload_file(file_path)
                        file_tokens.append(file_token)
                    final_record_dict[key] = [{"file_token": i} for i in file_tokens]
                else:
                    final_record_dict[key] = str(value)
            pass

        # 打印出错误的columns，将不添加
        for name in error_columns_names:
            warnings.warn("'{}' column value is unvalid, not add".format(name))

        response = self.add_record({"fields": final_record_dict})
        return response


class FeishuDatabase(object):
    """飞书多维表格的接口，用于深度学习的实验记录
    """
    def __init__(self) -> None:
        """构造函数，但内部不构造，必须手动调用init函数完成对象的构造
        """
        super().__init__()
        self.table_api = None
        self.record_dict = {}  # 提供一份内部的记录dict，可在多处地方调用更新，最后形成一份完整的dict
        pass

    def init(self, user_id, app_id, app_secret, app_token, table_id):
        """手动调用的构造函数

        Args:
            user_id (str): 用户主体的ID，在管理后台->组织架构->成员->点击自己的头像
            app_id (str): 创建应用的app id
            app_secret (str): 创建应用的app secret
            app_token (str): 需要操作的多维表格的token
            table_id (str): 需要操作的数据表token
        """
        self.table_api = MultiDimDocmentApi(user_id, app_id, app_secret, app_token, table_id)
        pass

    def record_add_keys(self, record_dict):
        """向内部的record dict添加记录的内容

        Args:
            record_dict (dict): 需要新增的记录的内容
        """
        for key, value in record_dict.items():
            if key in self.record_dict.keys():
                if isinstance(value, list) or isinstance(self.record_dict[key], list):
                    if not isinstance(self.record_dict[key], list):
                        self.record_dict[key] = [self.record_dict[key]]
                    if not isinstance(value, list):
                        value = [value]
                    self.record_dict[key].extend(value)
                else:
                    warnings.warn("record key '{}' has been record, it will be overrided.".format(key))
                    self.record_dict[key] = value
            else:
                self.record_dict[key] = value
        pass

    def record_push(self, exp_name, record_dict=None):
        """push record dict到飞书多维表格，如果不提供现成的record_dict就用内部的record_dict

        Args:
            exp_name (str): 实验名称
            record_dict (dict, optional): 需要push的record. Defaults to None.
        """
        if self.table_api is None:
            warnings.warn("必须先初始化database!")
            return 
        if record_dict is None:
            record_dict = self.record_dict
            pass
        final_dict = record_dict.copy()
        final_dict["实验记录"] = exp_name
        self.table_api.smart_add_record(final_dict)
        pass


feishu_database = FeishuDatabase()


if __name__ == "__main__":
    with open('feishu_app.json','r',encoding='utf8') as fp:
        json_data = json.load(fp)

    api = MultiDimDocmentApi(json_data["user_id"], json_data["app_id"], json_data["app_secret"], json_data["app_token"], json_data["table_id"])
    
    # test_new_record_1 = {
    #     "实验记录": "smart new record 测试",
    #     "多选": [1, 2, 3], 
    #     "单选": 1,
    # }
    # test_new_record_2 = {
    #     "实验记录": "smart new record 测试",
    #     "多选": [0.1, 0.2, 0.3], 
    #     "单选": 0.1
    # }
    # test_new_record_3 = {
    #     "实验记录": "smart new record 测试",
    #     "附件": "results.csv"
    # }

    # response = api.smart_add_record(test_new_record_1)
    # response = api.smart_add_record(test_new_record_2)
    # response = api.smart_add_record(test_new_record_3)
    pass