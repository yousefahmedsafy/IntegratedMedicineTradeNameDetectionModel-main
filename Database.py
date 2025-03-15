import pymssql




def fetch_medicines():    
        connectionString = pymssql.connect(
        server='AkremPlatform.mssql.somee.com',
        database='AkremPlatform',
        user='eslamfighting_SQLLogin_1',
        password='rjintm3u5f')

        cursor = connectionString.cursor()
        cursor.execute("SELECT Id,Name FROM MedicineCategories")
        medicines =  cursor.fetchall()
        connectionString.close()
        return medicines
