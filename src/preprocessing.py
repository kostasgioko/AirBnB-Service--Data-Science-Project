import pandas as pd

def drop_unused_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unused columns:
    - Drop features with useless information.
    - Drop features with correletion less than 0.04 with the target feature.

    Input parameter: data - pd.DataFrame
    Returns: The DataFrame after dropping the described columns.
    """
    # Features to drop.
    to_drop = ['id', 'listing_url', 'scrape_id', 'last_scraped', 'name', 'description', 'neighborhood_overview', 'picture_url', 'host_id', 'host_url',
        'host_name', 'host_location', 'host_about', 'host_acceptance_rate', 'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood',
        'host_total_listings_count', 'host_verifications', 'neighbourhood', 'neighbourhood_group_cleansed', 'property_type', 'bedrooms', 'beds',
        'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm',
        'maximum_nights_avg_ntm', 'calendar_updated', 'calendar_last_scraped', 'first_review', 'last_review', 'license', 'calculated_host_listings_count',
        'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms']
    
    data_copy = data.drop(labels = to_drop, axis = 1)

    return data_copy


def handle_missing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values:
    - Rows with missing values in 'bathrooms_text', 'reviews_per_month',
      'host_response_time' and'host_response_rate' are filled with zeros.
    - If any rows have missing data on the host, drop them.

    Input parameter: data - pd.DataFrame
    Returns: The DataFrame after handling the missing values.
    """

    data_copy = data.copy()

    # Fill missing values.
    data_copy.bathrooms_text.fillna(value = '0 baths', inplace = True)
    data_copy.reviews_per_month.fillna(value = 0, inplace = True)
    data_copy.host_response_time.fillna(value = "0", inplace = True)
    data_copy.host_response_rate.fillna(value = '0%', inplace = True)

    # These have no data on the host, drop them.
    no_host = data_copy[ data_copy.host_since.isna() ].index
    no_host_rows = data_copy.loc[no_host]
    data_copy.drop(no_host_rows.index, inplace = True)
    data_copy.reset_index(drop = True, inplace = True)

    return data_copy


def encode_variables(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encode variables:
    - The 'bathrooms_text' column is split into two new columns. 'Bathrooms'
      holding the total number of bathrooms and 'shared_bath' haviing a boolean
      value indicating whether the listing has a shared bath or not.
    - The 'amenities' column is first turned into a proper list, and then we
      keep the total number of amenities as the feature.
    - From the 'host_since' column, we only keep the year since the host was 
      registered.
    - For the 'host_response_rate' we only keep the numeric part as the feature.
    - The 'price' target column is cleaned and placed at the end of the dataframe.
    - For the 'room_type' and 'host_response_time' columns we use ordinal encoding.
    - The columns with boolean values are converted to 0 and 1 values.
    - For the 'neighbourhood_cleansed' column we use the amount of times the
      neighbourhood of the listing has appeared in total, as the feature.

    Input parameter: data - pd.DataFrame
    Returns: The DataFrame after encoding all variables.
    """

    data_copy = data.copy()

    # Special encodings first.

    # Bathroom encoding.
    # Create two new columns.
    bathrooms = []                  # 'bathrooms' will hold the amount of bathrooms the listing has. 
    shared_bath = []                # 'shared_bath' for if the bathrooms are shared.
    
    for text in data_copy.bathrooms_text:
        text = text.lower()
    
        if 'shared' in text:
            shared_bath.append(1)
        else:
            shared_bath.append(0)

        half_flag = False
        if 'half-bath' in text:
            half_flag = True
        
        text = text.split()
        
        try:
            baths = float(text[0])
            if half_flag:
                baths += 0.5
            bathrooms.append(baths)        
        except:
            bathrooms.append(0)
        
    data_copy.bathrooms = bathrooms
    data_copy.insert(15, 'shared_bath', shared_bath)

    data_copy.drop("bathrooms_text", axis = 1,inplace=True)



    # Amenities encoding.
    amenities = data_copy.amenities

    cleaned_amenities = []                                  # Initialize list for the amenities of all listings.
    for i in range(len(amenities)):                         # For each listing.
        current = amenities[i]                              # Get its amenities.

        current = current[1:-1].split(sep = ',')            # Remove leading and trailing bracket from string, and separate it by the commas.

        cleaned = []
        for amenity in current:                             # For each amenity in the list.
            amenity = amenity.replace('"', '')              # Remove double quote characters.
            amenity = amenity.lstrip()                      # Remove leading white space.
            
            cleaned.append(amenity)                         # Add amenity to the cleaned list.

        cleaned_amenities.append(cleaned)                   # Add list to overall list.
        
    amenities = pd.Series(cleaned_amenities)                # Create a Series of the list.

    data_copy.amenities = amenities                        # Replace amenities column in dataframe.

    data_copy.amenities = data_copy.amenities.apply(lambda x : len(x))



    # Host_since encoding.
    # Keep only the year.
    data_copy.host_since = pd.to_datetime(data_copy.host_since)
    data_copy.host_since = data_copy.host_since.apply(lambda x: x.year)


    # Host_response_rate encoding.
    # Keep the numeric part.
    data_copy.host_response_rate = data_copy.host_response_rate.apply(lambda x: float(x.split('%')[0]))



    # Price target column cleanup and place it in the end.
    data_copy.price = data_copy.price.apply(lambda x : float(x[1:].replace(',', '')))
    data_copy.insert(len(data_copy.columns), "target", data_copy.price)
    data_copy.drop("price", axis = 1, inplace = True)



    # Rest of the encodings.

    # Define mappings.
    room_type_map = {"Entire home/apt" : 3, "Private room" : 2,"Hotel room" : 1,"Shared room" : 0}
    host_response_time_map = {"within an hour" : 4,"within a few hours" : 3,"within a day" : 2,"a few days or more" : 1, "0" : 0}
    boolean_map = {'t' : 1,'f' : 0}
    neighbourhood_cleansed_map = data_copy.neighbourhood_cleansed.value_counts().to_dict()

    # Apply mappings.
    data_copy.room_type = data_copy.room_type.map(room_type_map)
    data_copy.host_response_time = data_copy.host_response_time.map(host_response_time_map)
    data_copy.instant_bookable = data_copy.instant_bookable.map(boolean_map)
    data_copy.has_availability = data_copy.has_availability.map(boolean_map)
    data_copy.host_is_superhost = data_copy.host_is_superhost.map(boolean_map)
    data_copy.host_has_profile_pic = data_copy.host_has_profile_pic.map(boolean_map)
    data_copy.host_identity_verified = data_copy.host_identity_verified.map(boolean_map)
    data_copy.neighbourhood_cleansed = data_copy.neighbourhood_cleansed.map(neighbourhood_cleansed_map)

    
    return data_copy


def run_preprocessing_pipeline(data: pd.DataFrame) -> pd.DataFrame:
    """
    Run preprocessing pipeline. Steps:
    - Drop unused columns.
    - Handle missing values.
    - Encode features.

    Input parameter: data - pd.DataFrame
    Returns: The DataFrame after the preprocessing pipeline.
    """

    data = drop_unused_columns(data)
    data = handle_missing(data)
    data = encode_variables(data)

    return data


if __name__ == '__main__':
    pass