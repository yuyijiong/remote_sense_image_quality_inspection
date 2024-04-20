# coding=gbk
import shapefile


def write_shp(target, shape_type, geo_data, field_names, field_types, field_sizes, field_decimals, field_values):
    """
    生成的文件名，自动生成.shp .dbf .shx
    target = "D:\\test"

    '''
    几何类型
    NULL = 0
    POINT = 1
    POLYLINE = 3
    POLYGON = 5
    MULTIPOINT = 8
    POINTZ = 11
    POLYLINEZ = 13
    POLYGONZ = 15
    MULTIPOINTZ = 18
    POINTM = 21
    POLYLINEM = 23
    POLYGONM = 25
    MULTIPOINTM = 28
    MULTIPATCH = 31
    '''
    shapeType = 1

    几何数据
    points = [[1,1],[2,2]]
    polylines = [[[[1,1],[2,2]],[[2,2],[3,3]]],[[[1,2],[2,4]]]]
    polygons = [[[[1,1],[2,2],[3,1]],[[3,1],[2,2],[3,2]]],[[[1,2],[2,3],[1,3]]]]
    pointzs = [[1,1,1],[2,2,2]]
    polylinezs = [[[[1,1,1],[2,2,2]],[[2,2,3],[3,3,4]]],[[[1,2,5],[2,4,6]]]]
    polygonzs = [[[[1,1,1],[2,2,2],[3,1,3]],[[3,1,4],[2,2,5],[3,2,6]]],[[[1,2,7],[2,3,8],[1,3,9]]]]

    属性名称
    field_names = ['test1','test2','test3']

    '''
    属性类型
    “C”：字符，文字。
    “N”：数字，带或不带小数。
    “F”：浮动（与“N”相同）。
    “L”：逻辑，表示布尔值True / False值。
    “D”：日期。
    “M”：备忘录，在GIS中没有意义，而是xbase规范的一部分。
    '''
    field_types = ['C','N','F']

    字段长度，默认50
    field_sizes = [50,50,50]

    字段精度，默认为0
    field_decimals = [0,0,2]

    属性值
    field_values = [['test',1,1.111],['测试',2,2.226]]
    """
    try:
        w = shapefile.Writer(target=target, shapeType=shape_type, autoBalance=1,encoding='GBK')
        # 添加属性
        for idx_field in range(len(field_names)):
            w.field(field_names[idx_field], field_types[idx_field], field_sizes[idx_field], field_decimals[idx_field])
        for idx_record in range(len(geo_data)):
            # 写出几何数据
            if shape_type == 1:
                w.point(*geo_data[idx_record])
            elif shape_type == 3:
                w.line(geo_data[idx_record])
            elif shape_type == 5:
                w.poly(geo_data[idx_record])
            elif shape_type == 11:
                w.pointz(*geo_data[idx_record])
            elif shape_type == 13:
                w.linez(geo_data[idx_record])
            elif shape_type == 15:
                w.polyz(geo_data[idx_record])
            else:
                continue

            # 写出属性数据
            record = []
            for idx_field in range(len(field_names)):
                record.append(field_values[idx_record][idx_field])
            w.record(*record)
        return "success"
    except shapefile.ShapefileException as e:
        return repr(e)

if __name__ == "__main__":
    field_names = ['test1', 'test2', 'test3']
    field_types = ['C', 'N', 'F']
    field_sizes = [50, 50, 50]
    field_decimals = [0, 0, 2]
    field_values = [['test', 1, 1.111], ['测试', 2, 2.226]]

    target = "D:\\data\\shp\\point"
    shape_type = 1
    points = [[1, 1], [2, 2]]
    print(write_shp(target, shape_type, points, field_names, field_types, field_sizes, field_decimals, field_values))


    target = "D:\\data\\shp\\polygon"
    shape_type = 5
    polygons = [[[[1, 1], [2, 2], [3, 1]], [[3, 1], [2, 2], [3, 2]]], [[[1, 2], [2, 3], [1, 3]]]]
    print(write_shp(target, shape_type, polygons, field_names, field_types, field_sizes, field_decimals, field_values))


