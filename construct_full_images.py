from common import *

class FullImageBuilder:

    def __init__(self) -> None:
        
        # Load the training data
        self.data = pd.read_csv(TRAINING_DATA_CSV)

    def select_relevant_data(self, td: pd.DataFrame, year, location, experiment, timepoint):
        td = td.loc[(td["year"]==year)&(td["location"]==location)\
                    &(td["timepoint"]==timepoint)&(td["experiment"]==experiment)].reset_index()
        
        if (td.shape[0] == 0):
            print("Requested data not found.")
            return None, None

        # Find the field indices
        field_index = ["row", "column"]
        unique_plot_coordinates = td[field_index]
        unique_plot_coordinates = unique_plot_coordinates.drop_duplicates().sort_values(by=field_index)
        td = td.iloc[unique_plot_coordinates.index]

        # Calculate the discrete dimensions of the observable plots
        # x, y need to be swapped here row==y, column==x

        td["field_y"] = td["row"] - td["row"].min()
        td["field_x"] = td["column"] - td["column"].min()
        field_dims = (td["field_x"].max(), td["field_y"].max())

        print("[INFO] Field Grid: x:{} y:{}".format(field_dims[0], field_dims[1]))
        return td, field_dims

    def load_images(self, td, image_type="sat"):
        if image_type not in ["sat", "uav"]:
            print("Unkown image type: {}".format(image_type))
            quit(1)

        single_image_dims = [0, 0]
        image_data = {}

        for image_file, x, y in zip(td["image_path"], td["field_x"], td["field_y"]):
            
            # Harmonize datastructures
            if image_type == "sat":
                image = SatImage(image_file).get_rgb()
                image_dims = image.shape[:2]
            else:
                uav_file = get_corresponding_uav_file(image_file)
                if uav_file is None:
                    continue
                image = cv2.imread(uav_file)
                image_dims = image.shape[:2]
            
            # Save the image data
            image_data[(x,y)] = image
            
            # Check for max dimensions
            for dim in (0, 1):
                single_image_dims[dim] = image_dims[dim] if image_dims[dim] > single_image_dims[dim] else single_image_dims[dim]

        print("[INFO] Single image dimensions: {}".format(single_image_dims))
        return image_data, single_image_dims

    def stitch_images(self, image_data, field_dims, single_image_dims, border_size=0):
        full_image_dims = (field_dims[0] * (single_image_dims[0]+border_size) + border_size,
                           field_dims[1] * (single_image_dims[1]+border_size) + border_size)

        frame = np.zeros((full_image_dims[0], full_image_dims[1], 3), dtype=np.uint8)

        for x in range(0, field_dims[0]):
            for y in range(0, field_dims[1]):
                if (x,y) not in image_data.keys():
                    continue
                image = image_data[(x,y)]
                x_start = border_size + x * (single_image_dims[0]+border_size)
                y_start = border_size + y * (single_image_dims[1]+border_size)
                frame[x_start:x_start+image.shape[0], y_start:y_start+image.shape[1]] = image

        return frame

    def construct_image(self, year, location, experiment, timepoint, border_size=2):
        
        # Find all the relevant entries in the data set
        td, field_dims = self.select_relevant_data(self.data, year, location, experiment, timepoint)

        # Load images into memory
        print("[INFO] Loading images.")
        
        sat_image_data, sat_image_dims = self.load_images(td, image_type="sat")
        uav_image_data, uav_image_dims = self.load_images(td, image_type="uav")
       
        sat_frame = self.stitch_images(sat_image_data, field_dims, sat_image_dims, border_size=border_size)
        
        td["yield_norm"] = (td["yield"] - self.data["yield"].min()) / (self.data["yield"].max() - self.data["yield"].min())
        #yield_rgb = scalar_to_rgb(yield_data_norm)
        
        for x in range(0, field_dims[0]):
            for y in range(0, field_dims[1]):
                if (x,y) not in sat_image_data.keys():
                    continue
                yield_norm = td.loc[(td["field_x"]==x)&(td["field_y"]==y), "yield_norm"].values
                rect_color = scalar_to_rgb(yield_norm)
                rect_a = (border_size//2 + y * (sat_image_dims[1]+border_size), border_size//2 + x * (sat_image_dims[0]+border_size))
                rect_b = (rect_a[0] + sat_image_dims[1] + border_size//2, rect_a[1] + sat_image_dims[0] + border_size//2)
                
                cv2.rectangle(sat_frame, rect_a, rect_b, (0, int(rect_color[0][1]), int(rect_color[0][0])),
                              thickness=border_size//2)
        
        cv2.imwrite("full_images/sat_full_{}_{}_{}_{}.bmp".format(year, location, experiment, timepoint), sat_frame)
       
        if(len(uav_image_data.keys())>0):
            uav_frame = self.stitch_images(uav_image_data, field_dims, uav_image_dims, border_size=0)
            cv2.imwrite("full_images/uav_full_{}_{}_{}_{}.bmp".format(year, location, experiment, timepoint), uav_frame)

    def contruct_all(self):
        td = self.data[["year", "location", "experiment", "timepoint"]]
        td = td.drop_duplicates().reset_index()
        for row_idx in range(td.shape[0]):

            year = td.iloc[row_idx]["year"]
            location = td.iloc[row_idx]["location"]
            experiment = td.iloc[row_idx]["experiment"]
            timepoint = td.iloc[row_idx]["timepoint"]
            print("[INFO] Constructing image for {} {} {} {}".format(year, location, experiment, timepoint))
            
            self.construct_image(year, location, experiment, timepoint)

if __name__ == "__main__":
    fib = FullImageBuilder()
    fib.contruct_all()