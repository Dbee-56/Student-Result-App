import streamlit as st
import pandas as pd
import os
from io import BytesIO

NUM_MATHS = 5

# ------------------------------
# Utility Functions
# ------------------------------
def load_file(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()  
    if ext == ".csv":
        return pd.read_csv(uploaded_file)
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(uploaded_file)
    elif ext == ".json":
        return pd.read_json(uploaded_file)
    elif ext == ".parquet":
        return pd.read_parquet(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def calDiv(per):
    if per>=60 and per<=100:
        return "I"
    elif per>=45 and per<60:
        return "II"
    elif per>=33 and per <45:
        return "III"
    else:
        return "F"

def calGrade(per):
    if per>=81 and per<=100:
        return 'A'
    elif per>=61 and per<81:
        return 'B'
    elif per>=41 and per<61:
        return 'C'
    elif per>=33 and per<41:
        return 'D'
    elif per>=0 and per<33:
        return 'E'

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📊 Student Result Processing App")

st.markdown("Upload **exactly 3 CSV/XLSX files** (Science, Commerce, Arts).")

uploaded_files = st.file_uploader(
    "Upload your 3 files",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) == 3:
    try:
        # Load all 3 files
        df = load_file(uploaded_files[0])   # Science
        df1 = load_file(uploaded_files[1])  # Commerce
        df2 = load_file(uploaded_files[2])  # Arts

        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except Exception:
                pass

        for col in df1.columns:
            try:
                df1[col] = pd.to_numeric(df1[col], errors="ignore")
            except Exception:
                pass

        for col in df2.columns:
            try:
                df2[col] = pd.to_numeric(df2[col], errors="ignore")
            except Exception:
                pass

        # ------------------------------
        # Your Entire Script Logic
        # ------------------------------
        df["Total"]=df[["Hindi", "English", "Physics", "Chemistry", "Maths","Bio"]].sum(axis=1)
        df1["Total"]=df1[["Hindi", "English", "Accounts", "Bussiness Studies", "Industrial Organization"]].sum(axis=1)
        df2["Total"]=df2[["Hindi", "English", "Political Science", "Geography", "History"]].sum(axis=1)

        df["Percentage"] = df["Total"].apply(lambda x:round((x/150)*100,2))
        df1["Percentage"] = df1["Total"].apply(lambda x:round((x/150)*100,2))
        df2["Percentage"] = df2["Total"].apply(lambda x:round((x/150)*100,2))

        df["Div"] = df["Percentage"].apply(calDiv)
        df1["Div"] = df1["Percentage"].apply(calDiv)
        df2["Div"] = df2["Percentage"].apply(calDiv)

        df["Grade"] = df["Percentage"].apply(calGrade)
        df1["Grade"] = df1["Percentage"].apply(calGrade)
        df2["Grade"] = df2["Percentage"].apply(calGrade)

        # ------------------ Summary Dictionaries ------------------
        sum_dict1 = {}
        sum_dict2 = {}
        sum_dict3 = {}

        sum_dict1["Criteria"] = ["Pass","Fail","Abs","Pass %"]
        sum_dict1["Hindi"] = [len(df[df["Hindi"]>=10]) ,len(df[df["Hindi"]<10]),int(df["Hindi"].isnull().sum()),round(len(df[df["Hindi"]>=10])/len(df)*100,2)]
        sum_dict1["English"] = [len(df[df["English"]>=10]),len(df[df["English"]<10]),int(df["English"].isnull().sum()),round(len(df[df["English"]>=10])/len(df)*100,2)]
        sum_dict1["Physics"] = [len(df[df["Physics"]>=10]),len(df[df["Physics"]<10]),int(df["Physics"].isnull().sum()),round(len(df[df["Physics"]>=10])/len(df)*100,2)]
        sum_dict1["Chemistry"] = [len(df[df["Chemistry"]>=10]),len(df[df["Chemistry"]<10]),int(df["Chemistry"].isnull().sum()),round(len(df[df["Chemistry"]>=10])/len(df)*100,2)]
        sum_dict1["Maths"] = [len(df[df["Maths"]>=10]),len(df[df["Maths"]<10]),int(df["Maths"][:NUM_MATHS].isnull().sum()),round(len(df[df["Maths"]>=10])/NUM_MATHS*100,2)]
        sum_dict1["Bio"] = [len(df[df["Bio"]>=10]),len(df[df["Bio"]<10]),int(df["Bio"][NUM_MATHS:].isnull().sum()),round(len(df[df["Bio"]>=10])/(len(df)-NUM_MATHS)*100,2)]

        sum_dict2["Criteria"] = ["Pass","Fail","Abs","Pass %"]
        sum_dict2["Hindi"] = [len(df1[df1["Hindi"]>=10]) ,len(df1[df1["Hindi"]<10]),int(df1["Hindi"].isnull().sum()),round(len(df1[df1["Hindi"]>=10])/len(df1)*100,2)]
        sum_dict2["English"] = [len(df1[df1["English"]>=10]),len(df1[df1["English"]<10]),int(df1["English"].isnull().sum()),round(len(df1[df1["English"]>=10])/len(df1)*100,2)]
        sum_dict2["Accounts"] = [len(df1[df1["Accounts"]>=10]),len(df1[df1["Accounts"]<10]),int(df1["Accounts"].isnull().sum()),round(len(df1[df1["Accounts"]>=10])/len(df1)*100,2)]
        sum_dict2["Bussiness Studies"] = [len(df1[df1["Bussiness Studies"]>=10]),len(df1[df1["Bussiness Studies"]<10]),int(df1["Bussiness Studies"].isnull().sum()),round(len(df1[df1["Bussiness Studies"]>=10])/len(df1)*100,2)]
        sum_dict2["Industrial Organization"] = [len(df1[df1["Industrial Organization"]>=10]),len(df1[df1["Industrial Organization"]<10]),int(df1["Industrial Organization"].isnull().sum()),round(len(df1[df1["Industrial Organization"]>=10])/len(df1)*100,2)]

        sum_dict3["Criteria"] = ["Pass","Fail","Abs","Pass %"]
        sum_dict3["Hindi"] = [len(df2[df2["Hindi"]>=10]) ,len(df2[df2["Hindi"]<10]),int(df2["Hindi"].isnull().sum()),round(len(df2[df2["Hindi"]>=10])/len(df2)*100,2)]
        sum_dict3["English"] = [len(df2[df2["English"]>=10]),len(df2[df2["English"]<10]),int(df2["English"].isnull().sum()),round(len(df2[df2["English"]>=10])/len(df2)*100,2)]
        sum_dict3["Political Science"] = [len(df2[df2["Political Science"]>=10]),len(df2[df2["Political Science"]<10]),int(df2["Political Science"].isnull().sum()),round(len(df2[df2["Political Science"]>=10])/len(df2)*100,2)]
        sum_dict3["Geography"] = [len(df2[df2["Geography"]>=10]),len(df2[df2["Geography"]<10]),int(df2["Geography"].isnull().sum()),round(len(df2[df2["Geography"]>=10])/len(df2)*100,2)]
        sum_dict3["History"] = [len(df2[df2["History"]>=10]),len(df2[df2["History"]<10]),int(df2["History"].isnull().sum()),round(len(df2[df2["History"]>=10])/len(df2)*100,2)]

        df_sum_scn = pd.DataFrame(sum_dict1)[["Criteria", "Hindi", "English", "Bio", "Physics", "Chemistry"]]
        df_sum_com = pd.DataFrame(sum_dict2)[["Criteria", "Hindi", "English", 'Accounts','Industrial Organization', 'Bussiness Studies']]
        df_sum_art = pd.DataFrame(sum_dict3)[["Criteria", "Hindi", "English", 'History', 'Political Science','Geography']]

        # ------------------ Final DF Logic (subject analysis) ------------------
        dfs = [df2, df1, df]
        ignore_cols = ["Name", "Gender", "Total", "Percentage","Div","Grade","Maths","Bio"]
        subjects = []
        for temp_df in dfs:
            subjects.extend([col for col in temp_df.columns if col not in ignore_cols])
        subjects = list(dict.fromkeys(subjects))
        
        dfs = [df2, df1, df]
        ignore_cols = ["Name", "Gender", "Total", "Percentage","Div","Grade","Maths","Bio"]
        subjects = []
        for temp_df in dfs:
            subjects.extend([col for col in temp_df.columns if col not in ignore_cols])
        subjects = list(dict.fromkeys(subjects))

        result = []
        for subj in subjects:
            total_boys = len(df[(df["Gender"]=="M")]) + len(df1[(df1["Gender"]=="M")]) + len(df2[(df2["Gender"]=="M")])
            total_girls = len(df[(df["Gender"]=="F")]) + len(df1[(df1["Gender"]=="F")]) + len(df2[(df2["Gender"]=="F")])

            if subj not in df2.columns:
                total_boys -= len(df2[(df2["Gender"]=="M")])
                total_girls -= len(df2[(df2["Gender"]=="F")])

            if subj not in df1.columns:
                total_boys -= len(df1[(df1["Gender"]=="M")])
                total_girls -= len(df1[(df1["Gender"]=="F")])

            if subj not in df.columns:
                total_boys -= len(df[(df["Gender"]=="M")])
                total_girls -= len(df[(df["Gender"]=="F")])
                    
            result.append([subj, total_boys, total_girls])

        result.append(["Maths", df[["Maths","Gender"]][:NUM_MATHS][df[["Maths","Gender"]][:NUM_MATHS]["Gender"]=="M"].shape[0], df[["Maths","Gender"]][:NUM_MATHS][df[["Maths","Gender"]][:NUM_MATHS]["Gender"]=="F"].shape[0]])
        result.append(["Bio", df[["Bio","Gender"]][NUM_MATHS:][df[["Bio","Gender"]][NUM_MATHS:]["Gender"]=="M"].shape[0], df[["Bio","Gender"]][NUM_MATHS:][df[["Bio","Gender"]][NUM_MATHS:]["Gender"]=="F"].shape[0]])

        final_df = pd.DataFrame(result, columns=["Subject", "Total Boys", "Total Girls"])

        abs_boy = []
        abs_girl = []
        for subj in subjects:
            total_boys = 0
            total_girls = 0

            if subj in df2.columns:
                total_boys += df2[(df2[subj].isna()) & (df2["Gender"]=="M")].shape[0]
                total_girls += df2[(df2[subj].isna()) & (df2["Gender"]=="F")].shape[0]

            if subj in df1.columns:
                total_boys += df1[(df1[subj].isna()) & (df1["Gender"]=="M")].shape[0]
                total_girls += df1[(df1[subj].isna()) & (df1["Gender"]=="F")].shape[0]

            if subj in df.columns:
                total_boys += df[(df[subj].isna()) & (df["Gender"]=="M")].shape[0]
                total_girls += df[(df[subj].isna()) & (df["Gender"]=="F")].shape[0]

            abs_boy.append(total_boys)
            abs_girl.append(total_girls)

        abs_boy.append(df[["Maths","Gender"]][:NUM_MATHS][(df[["Maths","Gender"]][:NUM_MATHS]["Gender"]=="M") & (df[["Maths","Gender"]][:NUM_MATHS]["Maths"].isna())].shape[0])
        abs_girl.append(df[["Maths","Gender"]][:NUM_MATHS][(df[["Maths","Gender"]][:NUM_MATHS]["Gender"]=="F") & (df[["Maths","Gender"]][:NUM_MATHS]["Maths"].isna())].shape[0])

        abs_boy.append(df[["Bio","Gender"]][NUM_MATHS:][(df[["Bio","Gender"]][NUM_MATHS:]["Gender"]=="M") & (df[["Bio","Gender"]][NUM_MATHS:]["Bio"].isna())].shape[0])
        abs_girl.append(df[["Bio","Gender"]][NUM_MATHS:][(df[["Bio","Gender"]][NUM_MATHS:]["Gender"]=="F") & (df[["Bio","Gender"]][NUM_MATHS:]["Bio"].isna())].shape[0])

        final_df["Absent Boys"] = abs_boy
        final_df["Absent Girls"] = abs_girl

        final_df['Present Boys'] = final_df['Total Boys'] - final_df['Absent Boys']
        final_df['Present Girls'] = final_df['Total Girls'] - final_df['Absent Girls']

        final_df = final_df[['Subject', 'Total Boys', 'Total Girls', 'Present Boys', 'Present Girls', 
                'Absent Boys', 'Absent Girls']]

        pass_boy = []
        pass_girl = []
        for subj in subjects:
            total_boys = 0
            total_girls = 0
            if subj in df2.columns:
                total_boys += len(df2[(df2[subj]>=10)&(df2["Gender"]=="M")])
                total_girls += len(df2[(df2[subj]>=10)&(df2["Gender"]=="F")])

            if subj in df1.columns:
                total_boys += len(df1[(df1[subj]>=10)&(df1["Gender"]=="M")])
                total_girls += len(df1[(df1[subj]>=10)&(df1["Gender"]=="F")])

            if subj in df.columns:
                total_boys += len(df[(df[subj]>=10)&(df["Gender"]=="M")])
                total_girls += len(df[(df[subj]>=10)&(df["Gender"]=="F")])

            pass_boy.append(total_boys)
            pass_girl.append(total_girls)

        pass_boy.append(len(df[(df["Maths"]>=10)&(df["Gender"]=="M")]))
        pass_girl.append(len(df[(df["Maths"]>=10)&(df["Gender"]=="F")]))

        pass_boy.append(len(df[(df["Bio"]>=10)&(df["Gender"]=="M")]))
        pass_girl.append(len(df[(df["Bio"]>=10)&(df["Gender"]=="F")]))

        final_df["Pass Boys"] = pass_boy
        final_df["Pass Girls"] = pass_girl

        fail_boy = []
        fail_girl = []
        for subj in subjects:
            total_boys = 0
            total_girls = 0
            if subj in df2.columns:
                total_boys += len(df2[(df2[subj]<10)&(df2["Gender"]=="M")])
                total_girls += len(df2[(df2[subj]<10)&(df2["Gender"]=="F")])

            if subj in df1.columns:
                total_boys += len(df1[(df1[subj]<10)&(df1["Gender"]=="M")])
                total_girls += len(df1[(df1[subj]<10)&(df1["Gender"]=="F")])

            if subj in df.columns:
                total_boys += len(df[(df[subj]<10)&(df["Gender"]=="M")])
                total_girls += len(df[(df[subj]<10)&(df["Gender"]=="F")])

            fail_boy.append(total_boys)
            fail_girl.append(total_girls)

        fail_boy.append(len(df[(df["Maths"]<10)&(df["Gender"]=="M")]))
        fail_girl.append(len(df[(df["Maths"]<10)&(df["Gender"]=="F")]))

        fail_boy.append(len(df[(df["Bio"]<10)&(df["Gender"]=="M")]))
        fail_girl.append(len(df[(df["Bio"]<10)&(df["Gender"]=="F")]))

        final_df["Fail Boys"] = fail_boy
        final_df["Fail Girls"] = fail_girl

        def temp(val, face, gender="M"):
            counts_total = {}
            for dframe in (df2, df1, df):
                if val in dframe.columns:
                    counts = (dframe[val][dframe["Gender"] == gender]
                            .apply(lambda x: calDiv(x / 30 * 100))
                            .value_counts().to_dict())
                    # Merge counts by summing values for the same key
                    for k, v in counts.items():
                        counts_total[k] = counts_total.get(k, 0) + v
            return counts_total.get(face, 0)  # Return combined count for face, else 0

        final_df["1st Boys"] = final_df['Subject'].apply(lambda x: temp(x,"I"))
        final_df["1st Girls"] = final_df['Subject'].apply(lambda x: temp(x,"I","F"))

        final_df["2nd Boys"] = final_df['Subject'].apply(lambda x: temp(x,"II"))
        final_df["2nd Girls"] = final_df['Subject'].apply(lambda x: temp(x,"II","F"))

        final_df["3rd Boys"] = final_df['Subject'].apply(lambda x: temp(x,"III"))
        final_df["3rd Girls"] = final_df['Subject'].apply(lambda x: temp(x,"III","F"))

        final_df["Percentage Pass"] = round(((final_df["Pass Boys"]+final_df["Pass Girls"])/(final_df["Total Boys"]+final_df["Total Girls"]))*100,1)

        def CountGrade(sub,face):
            counts_total = {}
            for dframe in (df2, df1, df):
                if sub in dframe.columns:
                    counts = (dframe[sub].apply(lambda x: calGrade(x / 30 * 100)).value_counts().to_dict())
                    if sub not in ['Maths','Bio']:
                        if 'E' in counts:
                            counts['E'] += dframe[sub][dframe[sub].isna()].shape[0]
                        else:
                            counts['E'] = dframe[sub][dframe[sub].isna()].shape[0]
                    else:
                        if sub=="Maths":
                            if 'E' in counts:
                                counts['E'] += df[sub][:NUM_MATHS][df[sub][:NUM_MATHS].isna()].shape[0]
                            else:
                                counts['E'] = df[sub][:NUM_MATHS][df[sub][:NUM_MATHS].isna()].shape[0]  
                        else:
                            if 'E' in counts:
                                counts['E'] += df[sub][NUM_MATHS:][df[sub][NUM_MATHS:].isna()].shape[0]
                            else:
                                counts['E'] = df[sub][NUM_MATHS:][df[sub][NUM_MATHS:].isna()].shape[0] 
                    for k, v in counts.items():
                        counts_total[k] = counts_total.get(k, 0) + v
            return counts_total.get(face, 0) 

        final_df["A"] = final_df["Subject"].apply(lambda x:CountGrade(x,'A'))
        final_df["B"] = final_df["Subject"].apply(lambda x:CountGrade(x,'B'))
        final_df["C"] = final_df["Subject"].apply(lambda x:CountGrade(x,'C'))
        final_df["D"] = final_df["Subject"].apply(lambda x:CountGrade(x,'D'))
        final_df["E"] = final_df["Subject"].apply(lambda x:CountGrade(x,'E'))

        final_df["A%"] = round((final_df["A"]/(final_df["Total Boys"]+final_df["Total Girls"]))*100,1)
        final_df["B%"] = round((final_df["B"]/(final_df["Total Boys"]+final_df["Total Girls"]))*100,1)
        final_df["C%"] = round((final_df["C"]/(final_df["Total Boys"]+final_df["Total Girls"]))*100,1)
        final_df["D%"] = round((final_df["D"]/(final_df["Total Boys"]+final_df["Total Girls"]))*100,1)
        final_df["E%"] = round((final_df["E"]/(final_df["Total Boys"]+final_df["Total Girls"]))*100,1)

        # … (Due to space, keeping this section intact – everything from your original script goes here including absent/present, pass/fail, grades, percentage pass, etc.) …

        # ------------------ Write Excel ------------------
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Student_scn", index=False)
            df_sum_scn.to_excel(writer, sheet_name="Summary_scn", index=False)
            df1.to_excel(writer, sheet_name="Student_com", index=False)
            df_sum_com.to_excel(writer, sheet_name="Summary_com", index=False)
            df2.to_excel(writer, sheet_name="Student_art", index=False)
            df_sum_art.to_excel(writer, sheet_name="Summary_art", index=False)
            final_df.to_excel(writer, sheet_name="Exsi", index=False)

        st.success("✅ Processing complete!")
        st.download_button(
            label="📥 Download Final Excel",
            data=output.getvalue(),
            file_name="students_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"⚠️ Error while processing files: {e}")

elif uploaded_files and len(uploaded_files) != 3:
    st.warning("⚠️ Please upload exactly **3 CSV/XLSX files**.")


