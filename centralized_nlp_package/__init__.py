# In your package's __init__.py or utils module
def show_docs():
    from IPython.display import IFrame
    display(IFrame(src='https://yourdocumentationurl.com', width='100%', height='600px'))


# import centralized_nlp_package
# centralized_nlp_package.show_docs()

# [Click here to access the documentation](https://yourdocumentationurl.com)
