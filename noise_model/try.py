def filter_crfs(input_file, output_file, keyword='800'):
    """
    Filters out CRFs whose names contain the specified keyword and writes the rest to a new file.
    
    Args:
    - input_file (str): Path to the input file with CRFs.
    - output_file (str): Path to the output file to save the filtered CRFs.
    - keyword (str): Keyword to filter CRF names.
    """
    try:
        with open(input_file, 'r') as file:
            lines = file.readlines()

        # Prepare a list to store filtered lines
        filtered_lines = []

        # Process CRFs in groups of 6 lines
        for i in range(0, len(lines), 6):
            crf_name = lines[i].strip()  # CRF name is on the first line of each group
            print(crf_name)
            
        '''
            # If the CRF name does not contain the keyword, keep the group
            if keyword not in crf_name:
                filtered_lines.extend(lines[i:i+6])

        # Write the filtered CRFs to the output file
        with open(output_file, 'w') as file:
            file.writelines(filtered_lines)

        print(f"Filtered CRFs written to {output_file}")
        '''
    
    except FileNotFoundError:
        print(f"File not found: {input_file}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
input_file = 'dorfCurves_filtered.txt'
output_file = 'dorfCurves_filtered_new.txt'
filter_crfs(input_file, output_file)
