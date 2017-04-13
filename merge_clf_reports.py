import os
import personal_settings
from functools import reduce

root_path = "/home/nandane/Documents/Cours_ETS_MTL/LOG770_Intelligence_machine/LAB4/Processing_output_2"


def write_summary(path, files):
    output = open(path + "/" + "summary.txt", 'w')
    for file in files:
        print(path + "/" + file)
        input = open(path + "/" + file, 'r')
        output.write(str(file))
        output.write("\n\n")
        for line in input.readlines():
            if line.startswith("Classifications"):
                output.write(line)
        output.write("\n")
        output.write("-----------------------------------")
        output.write("\n")
    output.close()


def write_summary_for_all_outputs():
    for d, dd, files in os.walk(root_path):

        if d.endswith("evaluations"):
            print("d = " + str(d))
            print("files = " + str(files))
            write_summary(d, files)


def combine_summary_files():
    extraction_method = os.listdir(personal_settings.PATH)
    csv_file_name = "results.csv"
    csv_file = open(root_path + "/" + csv_file_name, 'w')
    dict = {}

    print(str(extraction_method))

    for d, dd, files in os.walk(root_path):
        if d.endswith("evaluations"):

            algorithm = d.split("/")[d.split("/").__len__() - 2].split("output_")[1]
            print("ALGO = " + algorithm)
            print(d)

            for file in files:
                if file.startswith("summary"):
                    summary_file = open(d + "/" + file, "r").readlines()

                    line_counter = 0
                    for line in summary_file:
                        line_counter += 1
                        if line.startswith("msd"):
                            e = line.split("_")[0].split(".txt")[0].upper()
                            index = extraction_method.index(e)
                            option = line.split("_")
                            if option.__len__() > 1:
                                option = "-".join(option[1:]).split(".")[0]

                            else:
                                option = None

                            method_name = algorithm
                            if option != None:
                                method_name = method_name + "_" + option

                            clf_value = summary_file[line_counter + 1].split(": 0.")[1][:2] + "%"
                            print("extraction_method = " + e)
                            print("method : " + method_name)
                            print("clf value = " + str(clf_value))

                            if dict.keys().__contains__(method_name):
                                dict[method_name][index] = clf_value
                            else:
                                dict[method_name] = ['0'] * extraction_method.__len__()
                                dict[method_name][index] = clf_value

    print(str(dict))
    csv_file.write("classifier," + ",".join(extraction_method) + "\n")
    for key in dict:
        csv_file.write(key + "," + ",".join(dict[key]) + "\n")
    csv_file.close()

write_summary_for_all_outputs()
combine_summary_files()