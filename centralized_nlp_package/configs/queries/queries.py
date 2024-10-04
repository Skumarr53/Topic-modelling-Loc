embeddingCreateInpQuery = """
                      SELECT 
                        FILT_DATA, 
                        ENTITY_ID, 
                        UPLOAD_DT_UTC, 
                        VERSION_ID, 
                        EVENT_DATETIME_UTC 
                      FROM 
                          EDS_PROD.QUANT.YUJING_CT_TL_STG_1 
                      WHERE 
                          EVENT_DATETIME_UTC >= "{min_date}" 
                          AND EVENT_DATETIME_UTC < "{max_date}"
                      """