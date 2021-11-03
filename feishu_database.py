from typing import final
import requests
import json


class MultiDimDocmentApi(object):
    """
    一次只能针对一个飞书一个多维表格的一个数据表进行操作
    """
    def __init__(self, app_id, app_secret, app_token, table_id) -> None:
        """

        Args:
            app_id (str): 创建应用的app id
            app_secret (str): 创建应用的app secret
            app_token (str): 需要操作的多维表格的token
            table_id (str): 需要操作的数据表token
        """
        super().__init__()
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
        response = requests.post(url, data=json.dumps(record_dict).encode("utf-8"), headers=header)
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
                elif exist_column_type == 4:
                    # 判断是否为列表，如果不是则无效
                    if isinstance(value, list):
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
                    else:
                        error_columns_names.append(key)
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
                else:
                    final_record_dict[key] = str(value)
            pass

        # 打印出错误的columns，将不添加
        for name in error_columns_names:
            print("'{}' column value is unvalid, not add".format(name))

        response = self.add_record({"fields": final_record_dict})
        return response


class FeishuDatabase(object):
    def __init__(self) -> None:
        super().__init__()
        self.table_api = None
        pass

    def init(self, app_id, app_secret, app_token, table_id):
        self.table_api = MultiDimDocmentApi(app_id, app_secret, app_token, table_id)
        pass

    def new_record(self, exp_name, record_dict):
        if self.table_api is None:
            print("必须先初始化database!")
            return 
        final_dict = record_dict.copy()
        final_dict["实验记录"] = exp_name
        self.table_api.smart_add_record(final_dict)
        pass

feishu_database = FeishuDatabase()

if __name__ == "__main__":
    with open('feishu_app.json','r',encoding='utf8') as fp:
        json_data = json.load(fp)

    api = MultiDimDocmentApi(json_data["app_id"], json_data["app_secret"], json_data["app_token"], json_data["table_id"])
    
    test_new_record_1 = {
        "实验记录": "smart new record 测试",
        "多选": [1, 2, 3], 
        "单选": 1,
    }
    test_new_record_2 = {
        "实验记录": "smart new record 测试",
        "多选": [0.1, 0.2, 0.3], 
        "单选": 0.1
    }
    test_new_record_3 = {
        "实验记录": "smart new record 测试",
        "多选": [2, 3, 5], 
        "单选": 12,
        "测试新增": [1, 2, 3]
    }

    response = api.smart_add_record(test_new_record_1)
    response = api.smart_add_record(test_new_record_2)
    response = api.smart_add_record(test_new_record_3)
    pass