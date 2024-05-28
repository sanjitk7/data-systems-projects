import flatbuffers
import pandas as pd
import struct
import time
import types

import numpy as np
from CS598MP3 import DataFrame
from CS598MP3 import DataFrameMetadata
from CS598MP3 import Column
from CS598MP3 import IntColumn
from CS598MP3 import FloatColumn
from CS598MP3 import StringColumn
from CS598MP3 import AnyColumn

import CS598MP3
import CS598MP3.DataFrame

# Your Flatbuffer imports here (i.e. the files generated from running ./flatc with your Flatbuffer definition)...

def to_flatbuffer(df: pd.DataFrame) -> bytearray:
    """
        Converts a DataFrame to a flatbuffer. Returns the bytearray of the flatbuffer.

        The flatbuffer should follow a columnar format as follows:
        +-------------+----------------+-------+-------+-----+----------------+-------+-------+-----+
        | DF metadata | col 1 metadata | val 1 | val 2 | ... | col 2 metadata | val 1 | val 2 | ... |
        +-------------+----------------+-------+-------+-----+----------------+-------+-------+-----+
        You are free to put any bookkeeping items in the metadata. however, for autograding purposes:
        1. Make sure that the values in the columns are laid out in the flatbuffer as specified above
        2. Serialize int and float values using flatbuffer's 'PrependInt64' and 'PrependFloat64'
            functions, respectively (i.e., don't convert them to strings yourself - you will lose
            precision for floats).

        @param df: the dataframe.
    """
    
    builder = flatbuffers.Builder(1024)
    
    # Create metadata for the DataFrame
    metadata_name_offset = builder.CreateString("CS598_MP3_Dataframe")
    DataFrameMetadata.DataFrameMetadataStart(builder)
    DataFrameMetadata.DataFrameMetadataAddName(builder, metadata_name_offset)
    DataFrameMetadata.DataFrameMetadataAddNumRows(builder, df.shape[0])
    DataFrameMetadata.DataFrameMetadataAddNumColumns(builder, df.shape[1])
    metadata_offset = DataFrameMetadata.DataFrameMetadataEnd(builder)

    columns_offsets = []    
    for col_name, col_data in reversed(list(df.items())):
        name_offset = builder.CreateString(col_name)
        
        if col_data.dtype == 'int64':
            IntColumn.IntColumnStartIntValuesVector(builder, len(col_data))
            for val in reversed(col_data):
                builder.PrependInt64(val)
            values_offset = builder.EndVector(len(col_data))

            IntColumn.IntColumnStart(builder)
            IntColumn.IntColumnAddIntValues(builder, values_offset)
            int_col_offset = IntColumn.IntColumnEnd(builder)
            column_type = AnyColumn.AnyColumn.IntColumn  # Union type identifier

        elif col_data.dtype == 'float64':
            FloatColumn.FloatColumnStartFloatValuesVector(builder, len(col_data))
            for val in reversed(col_data):
                builder.PrependFloat64(val)
            values_offset = builder.EndVector(len(col_data))

            FloatColumn.FloatColumnStart(builder)
            FloatColumn.FloatColumnAddFloatValues(builder, values_offset)
            float_col_offset = FloatColumn.FloatColumnEnd(builder)
            column_type = AnyColumn.AnyColumn.FloatColumn

        elif col_data.dtype == 'object':  # assuming string for object dtype
            strings = [builder.CreateString(str(x)) for x in col_data]
            StringColumn.StringColumnStartStringValuesVector(builder, len(strings))
            for s in reversed(strings):
                builder.PrependUOffsetTRelative(s)
            values_offset = builder.EndVector(len(col_data))

            StringColumn.StringColumnStart(builder)
            StringColumn.StringColumnAddStringValues(builder, values_offset)
            string_col_offset = StringColumn.StringColumnEnd(builder)
            column_type = AnyColumn.AnyColumn.StringColumn

        # Create the Column table
        Column.ColumnStart(builder)
        Column.ColumnAddName(builder, name_offset)
        
        # Set the type and value for the AnyColumn union
        if col_data.dtype == 'int64':
            Column.ColumnAddValuesType(builder, column_type)
            Column.ColumnAddValues(builder, int_col_offset)
        elif col_data.dtype == 'float64':
            Column.ColumnAddValuesType(builder, column_type)
            Column.ColumnAddValues(builder, float_col_offset)
        elif col_data.dtype == 'object':
            Column.ColumnAddValuesType(builder, column_type)
            Column.ColumnAddValues(builder, string_col_offset)
        column_offset = Column.ColumnEnd(builder)

        columns_offsets.append(column_offset)

    # Create the DataFrame
    DataFrame.DataFrameStartDfColumnsVector(builder, len(columns_offsets))
    for col in reversed(columns_offsets):
        builder.PrependUOffsetTRelative(col)
    columns_vector_offset = builder.EndVector(len(columns_offsets))

    DataFrame.DataFrameStart(builder)
    DataFrame.DataFrameAddDfMetadata(builder, metadata_offset)
    DataFrame.DataFrameAddDfColumns(builder, columns_vector_offset)
    data_frame_offset = DataFrame.DataFrameEnd(builder)

    builder.Finish(data_frame_offset)
    return builder.Output()


def fb_dataframe_head(fb_bytes: bytes, rows: int = 5) -> pd.DataFrame:
    """
        Returns the first n rows of the Flatbuffer Dataframe as a Pandas Dataframe
        similar to df.head(). If there are less than n rows, return the entire Dataframe.
        Hint: don't forget the column names!

        @param fb_bytes: bytes of the Flatbuffer Dataframe.
        @param rows: number of rows to return.
    """
    # Initialize the buffer and get the DataFrame
    df = DataFrame.DataFrame.GetRootAsDataFrame(fb_bytes, 0)

    # Initialize the dictionary to hold column data
    columns_data = {}
    num_columns = df.DfColumnsLength()

    for i in range(num_columns):
        column = df.DfColumns(i)
        col_name = column.Name().decode('utf-8')  # Decode the name of the column
        
        # print("column name: ", col_name)

        # Checking the type of values in the union
        column_type = column.ValuesType()
        if column_type == AnyColumn.AnyColumn().IntColumn:
            int_column = IntColumn.IntColumn()
            int_column.Init(column.Values().Bytes, column.Values().Pos)
            column_values = [int_column.IntValues(j) for j in range(min(rows, int_column.IntValuesLength()))]

        elif column_type == AnyColumn.AnyColumn().FloatColumn:
            float_column = FloatColumn.FloatColumn()
            float_column.Init(column.Values().Bytes, column.Values().Pos)
            column_values = [float_column.FloatValues(j) for j in range(min(rows, float_column.FloatValuesLength()))]

        elif column_type == AnyColumn.AnyColumn().StringColumn:
            string_column = StringColumn.StringColumn()
            string_column.Init(column.Values().Bytes, column.Values().Pos)
            column_values = [string_column.StringValues(j).decode('utf-8') for j in range(min(rows, string_column.StringValuesLength()))]

        # Store the extracted values in a dictionary using the column name as the key
        columns_data[col_name] = column_values

    # Create a Pandas DataFrame from the dictionary
    result_df = pd.DataFrame(columns_data)
    return result_df[result_df.columns[::-1]]



def fb_dataframe_group_by_sum(fb_bytes: bytes, grouping_col_name: str, sum_col_name: str) -> pd.DataFrame:
    """
        Applies GROUP BY SUM operation on the flatbuffer dataframe grouping by grouping_col_name
        and summing sum_col_name. Returns the aggregate result as a Pandas dataframe.

        @param fb_bytes: bytes of the Flatbuffer Dataframe.
        @param grouping_col_name: column to group by.
        @param sum_col_name: column to sum.
    """
    # Access the root of the DataFrame Flatbuffer
    df = DataFrame.DataFrame.GetRootAsDataFrame(fb_bytes, 0)

    # Initialize storage for column data
    group_by_data = []
    sum_data = []

    count = 1
    # Iterate over columns to extract the necessary data for 'grouping_col_name' and 'sum_col_name'
    for i in range(df.DfColumnsLength()):
        column = df.DfColumns(i)
        col_name = column.Name().decode('utf-8')

        # Check if this is the column we need for grouping or summing
        if col_name == grouping_col_name or col_name == sum_col_name:
            print("col found: ", col_name)
            count+=1
            # column_type = column.ValuesType()
            # if column_type == AnyColumn.AnyColumn().IntColumn:
            int_column = IntColumn.IntColumn()
            int_column.Init(column.Values().Bytes, column.Values().Pos)
            column_values = [int_column.IntValues(j) for j in range(int_column.IntValuesLength())]

            # Append to appropriate list
            if col_name == grouping_col_name:
                group_by_data = column_values
            elif col_name == sum_col_name:
                sum_data = column_values
        
        if count > 2:
            # print("both cols found!!!!")
            break
    
    
    part3start = time.time()
    # Create a DataFrame from the collected data
    temp_df = pd.DataFrame({grouping_col_name: group_by_data, sum_col_name: sum_data})

    # Perform the groupby and sum operation
    result_df = temp_df.groupby(grouping_col_name).agg({sum_col_name: 'sum'}).reset_index()
    part3time = time.time() - part3start
    

    print("part3time: ", part3time)
    return result_df.set_index(grouping_col_name)
    
    return pd.DataFrame()  # REPLACE THIS WITH YOUR CODE...


def fb_dataframe_map_numeric_column(fb_buf: memoryview, col_name: str, map_func: types.FunctionType) -> None:
    """
        Apply map_func to elements in a numeric column in the Flatbuffer Dataframe in place.
        This function shouldn't do anything if col_name doesn't exist or the specified
        column is a string column.

        @param fb_buf: buffer containing bytes of the Flatbuffer Dataframe.
        @param col_name: name of the numeric column to apply map_func to.
        @param map_func: function to apply to elements in the numeric column.
    """
    # YOUR CODE HERE...
    pass
    
