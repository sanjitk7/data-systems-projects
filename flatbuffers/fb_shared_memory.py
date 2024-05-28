import dill
import hashlib
import pandas as pd
import types
import pickle

from multiprocessing import shared_memory

from fb_dataframe import to_flatbuffer, fb_dataframe_head, fb_dataframe_group_by_sum, fb_dataframe_map_numeric_column


class FbSharedMemory:
    """
        Class for managing the shared memory for holding flatbuffer dataframes.
    """
    def __init__(self):
        self.shared_name = "CS598"
        try:
            self.df_shared_memory = shared_memory.SharedMemory(name=self.shared_name)
            self.offsets = pickle.loads(bytes(self.df_shared_memory.buf[:1024]).rstrip(b'\x00'))
            # Maintain the current data end offset, initialize if not present
            self.data_end = max(self.offsets.values(), default=1024) if self.offsets else 1024
        except FileNotFoundError:
            # Shared memory is not created yet, create it with size 200M.
            self.df_shared_memory = shared_memory.SharedMemory(name=self.shared_name, create=True, size=200000000)
            self.offsets = {}
            self.data_end = 1024  # Start after the offset dictionary space
            self.df_shared_memory.buf[:1024] = pickle.dumps(self.offsets).ljust(1024, b'\x00')

    def add_dataframe(self, name: str, df: pd.DataFrame) -> None:
        if name in self.offsets:
            print("Dataframe with this name already exists.")
            return

        # Serialize DataFrame to Flatbuffer
        fb_bytes = to_flatbuffer(df)
        new_offset = self.data_end
        required_size = new_offset + len(fb_bytes)

        # Ensure there is enough space left in the shared memory
        if required_size > self.df_shared_memory.size:
            raise MemoryError("Not enough shared memory available.")

        # Add the serialized dataframe to shared memory
        self.df_shared_memory.buf[new_offset:new_offset + len(fb_bytes)] = fb_bytes
        self.offsets[name] = new_offset
        self.data_end = new_offset + len(fb_bytes)  # Update the end of the data

        # Update the offsets dictionary in shared memory
        serialized_offsets = pickle.dumps(self.offsets).ljust(1024, b'\x00')
        self.df_shared_memory.buf[:1024] = serialized_offsets


    def _get_fb_buf(self, df_name: str) -> memoryview:
        """
            Returns the section of the buffer corresponding to the dataframe with df_name.
            Hint: get buffer section (fb_buf) holding the flatbuffer from shared memory.

            @param df_name: name of the Dataframe.
        """
        
        if df_name not in self.offsets:
            raise ValueError("Dataframe not found in shared memory.")
        start_offset = self.offsets[df_name]
        # Assume each FB ends where the next begins or at end of buffer, calculate the end offset if needed
        # For simplicity, we're not calculating end offsets here, assuming dataframes are appended sequentially
        return self.df_shared_memory.buf[start_offset:]  # This should ideally have a way to determine the end offset

    # Other methods remain unchanged

    def close(self) -> None:
        """
            Closes the managed shared memory.
        """
        try:
            self.df_shared_memory.close()
            if hasattr(self, 'df_shared_memory'):
                self.df_shared_memory.unlink()  # Optionally unlink
        except:
            pass


    def dataframe_head(self, df_name: str, rows: int = 5) -> pd.DataFrame:
        """
            Returns the first n rows of the Flatbuffer Dataframe as a Pandas Dataframe
            similar to df.head(). If there are less than n rows, returns the entire Dataframe.

            @param df_name: name of the Dataframe.
            @param rows: number of rows to return.
        """
        fb_bytes = bytes(self._get_fb_buf(df_name))
        return fb_dataframe_head(fb_bytes, rows)

    def dataframe_group_by_sum(self, df_name: str, grouping_col_name: str, sum_col_name: str) -> pd.DataFrame:
        """
            Applies GROUP BY SUM operation on the flatbuffer dataframe grouping by grouping_col_name
            and summing sum_col_name. Returns the aggregate result as a Pandas dataframe.
    
            @param df_name: name of the Dataframe.
            @param grouping_col_name: column to group by.
            @param sum_col_name: column to sum.
        """
        fb_bytes = bytes(self._get_fb_buf(df_name))
        return fb_dataframe_group_by_sum(fb_bytes, grouping_col_name, sum_col_name)

    def dataframe_map_numeric_column(self, df_name: str, col_name: str, map_func: types.FunctionType) -> None:
        """
            Apply map_func to elements in a numeric column in the Flatbuffer Dataframe in place.

            @param df_name: name of the Dataframe.
            @param col_name: name of the numeric column to apply map_func to.
            @param map_func: function to apply to elements in the numeric column.
        """
        fb_dataframe_map_numeric_column(self._get_fb_buf(df_name), col_name, map_func)


    def close(self) -> None:
        """
            Closes the managed shared memory.
        """
        try:
            self.df_shared_memory.close()
        except:
            pass