from lab4_backend import *

import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk

class LabGUI(tk.Frame):
    
    def __init__(self, parent):
        
        tk.Frame.__init__(self, parent)
        self.parent = parent
        
        parent.geometry('1450x700+50+50')
        parent.title('approximation')
                        
        self.init_feature_filenames()
        self.init_label_filename()
                
        self.init_choose_polynomial()
        self.init_create_pads()
        
        self.init_polynomial_degrees()
        
        self.init_y_coord()
        
        self.init_run_button()
        self.init_default_checkbutton()
        self.init_y_checkbutton()
        
        self.init_mult()

        self.init_all_labels()
        self.init_output_label()
        self.init_graph_canvas()
        
        self.bind_keyboard()
        
        self.image_ind = 0
        self.image_amount = 0
        
        self.first_run = True
        
    def init_label(self, text, column, row, columnspan=1, rowspan=1, height=1):
        
        label = tk.Label(self.parent, text=text, height=height)
        label.grid(column=column, row=row, columnspan=columnspan, rowspan=rowspan)
        
        
    def init_variable_label(self, variable, column, row, columnspan=1, rowspan=1):
        
        label = tk.Label(self.parent, textvariable=variable)
        label.grid(column=column, row=row, columnspan=columnspan, rowspan=rowspan)
        
        
    def init_entry(self, variable, text, column, row, columnspan=1, rowspan=1, width=4):
        
        entry = tk.Entry(
            self.parent, text=text,
            textvariable=variable, width=width)    
        
        entry.grid(column=column, row=row, columnspan=columnspan, rowspan=rowspan, sticky=tk.W)
    
    
    def init_label_with_entry(self, text, column, row, variable, columnspan=1, rowspan=1, entry_width=4):
        
        self.init_label(text=text, column=column, row=row)       
        self.init_entry(text=text, column=column+1, row=row, variable=variable, width=entry_width)
 

    def init_radiobutton(self, text, column, row, variable, value, columnspan=1):
        
        rdbutton = tk.Radiobutton(self.parent, text=text, value=value, variable=variable)
        rdbutton.grid(row=row, column=column, columnspan=columnspan, sticky=tk.W)
        
        
    def init_pads(self, row=1, column=1, width=0, height=0):
        
        label = tk.Frame(self.parent, width=width, height=height)
        label.grid(column=column, row=row)
    
    
    def init_button(self, text, column, row, width, height, columnspan=1, rowspan=1, command=None):
        butt = tk.Button(self.parent, text=text, width=width, height=height, command=command)
        butt.grid(column=column, row=row, columnspan=columnspan, rowspan=rowspan)
        
        
    def init_checkbutton(self, text, column, row, columnspan=1, rowspan=1, variable=None):
        butt = tk.Checkbutton(self.parent, text=text, variable=variable)
        butt.grid(column=column, row=row, columnspan=columnspan, rowspan=rowspan)
    
    #Buttons
    
    def init_run_button(self):
        self.init_button(
            text = 'Розрахувати', width=10, height=1, 
            column=11, row=6, rowspan=2, columnspan=2,
            command=self.run
        )
    
    
    def init_default_checkbutton(self):
        self.default_data = tk.BooleanVar(value=True)
        
        self.init_checkbutton(
            text = 'Стандартні значення', 
            column=1, row=4,
            variable=self.default_data
        )
        
    def init_y_checkbutton(self):
        
        self.normalize_y = tk.BooleanVar(value=True)
        
        self.init_checkbutton(
            text = 'Нормалізувати y на графіку', 
            column=11, row=5,
            variable=self.normalize_y
        )
    
    # Labels
    
    def init_create_pads(self):
        
        self.init_pads(column=2, width=35)
        self.init_pads(column=5, width=35)
        self.init_pads(column=7, width=20)
        self.init_pads(column=10, width=35)
        self.init_pads(row=7, height=30)
        self.init_pads(row=9, height=8)
    
    def init_all_labels(self):
                
        self.init_label(text = 'Вибірка', column=0, row=0,columnspan=2, rowspan=2)
        self.init_label(text = 'Поліноми', column=6, row=0, columnspan=4, height=2)
        self.init_label(text = 'Вид', column=6, row=1)
        self.init_label(text = 'Степінь', column=8, row=1)
        
        self.init_label(text = 'Результат', column=0, row=8, columnspan=6)
        self.init_label(text = 'Графік', column=7, row=8, columnspan=6)  
        
        self.err_output = tk.StringVar(value='Значення похибки')
        self.init_variable_label(variable=self.err_output, column=8, row=11, columnspan=6)
        
    
    def init_output_label(self):
        
        self.scrolled_text = ScrolledText(
            master=self.parent, wrap='word', width=80, height=24, font=('TkDefaultFont', 11))
        
        self.scrolled_text.grid(column=0, row=10, columnspan=7)
        self.set_output_text('Натисніть на кнопку "Розрахувати"')
        self.scrolled_text.config(state=tk.DISABLED)
    
    
    def init_graph_canvas(self):
        self.canvas_size = (700, 420)
        self.graph_canvas = tk.Canvas(self.parent, bg="#fff", height=self.canvas_size[1], width=self.canvas_size[0])
        self.graph_canvas.grid(column=8, row=10, columnspan=6, sticky=tk.NW)
    
    # RadioButtons   
    
    def init_choose_polynomial(self):
        self.polynomial_type = tk.StringVar(value='chebyshev_first')
        self.init_radiobutton(text = 'Чебишева 1-го порядку', row=2, column=6, value='chebyshev_first', variable=self.polynomial_type)
        self.init_radiobutton(text = 'Чебишева 2-го порядку', row=3, column=6, value='chebyshev_second', variable=self.polynomial_type)
        self.init_radiobutton(text = 'Лежандра', row=4, column=6, value='legendre', variable=self.polynomial_type)
        self.init_radiobutton(text = 'Лагерра', row=5, column=6, value='laguerre', variable=self.polynomial_type)
        self.init_radiobutton(text = 'Ерміта', row=6, column=6, value='hermite', variable=self.polynomial_type)
         
    # Labels with entries
    
    def init_samples_length(self):
        
        self.samples_length = tk.IntVar()
        self.init_label_with_entry(
            text='Розмір вибірки', column=0, row=2,
            variable=self.samples_length)
        
    def init_feature_filenames(self):
        self.feature_filenames = tk.StringVar(value='Data/X{}_Warn2.txt')
        self.init_label_with_entry(
            text='Файли зі значеннями x', column=0, row=2,
            variable=self.feature_filenames, entry_width=15)
        
    def init_label_filename(self):
        self.label_filename = tk.StringVar(value='Data/Y_Warn2.txt')
        self.init_label_with_entry(
            text='Файл зі значеннями y', column=0, row=3,
            variable=self.label_filename, entry_width=15) 
    
    
    def init_polynomial_degrees(self):
        self.polynomial_degrees = [tk.IntVar(value=3) for _ in range(3)]
        
        for ind, var in enumerate(self.polynomial_degrees):  
            self.init_label_with_entry(
                text='Степінь x' + str(ind+1), column=8, row=ind+2,
                variable=var)
    
    def init_y_coord(self):
        self.y_coord = tk.IntVar(value=1)
        self.init_label_with_entry(
            text='Координата y', column=11, row=4,
            variable=self.y_coord)
        
    def init_mult(self):
        self.init_label(text = 'Модель', column=1, row=5, columnspan=3)
        
        self.mult = tk.BooleanVar(value=True)
        
        self.init_radiobutton(
            text = 'Адитивна', row=6, column=1, columnspan=2, 
            value=False, variable=self.mult)
        
        self.init_radiobutton(
            text = 'Мультиплікативна', row=6, column=2, columnspan=2,
            value=True, variable=self.mult) 
    
    
    def set_output_text(self, output):
        
        self.scrolled_text.config(state=tk.NORMAL)
        self.scrolled_text.delete(1.0, tk.END)
        self.scrolled_text.insert(1.0, output)
        self.scrolled_text.config(state=tk.DISABLED)
    
    
    def update_recursive(self):
        self.image_ind = (self.image_ind + 1) % self.image_amount
        self.graph_canvas.itemconfigure(self.canvas_image, image=self.graph_images[self.image_ind])
        self.parent.after(300, self.update_recursive)
    
    
    def update_graph(self):
        
        self.graph_images = [ImageTk.PhotoImage(
            Image.open(
                'graph_{0}.png'.format(image_ind + 1)).resize(self.canvas_size, Image.ANTIALIAS)) 
                             for image_ind in range(self.image_amount)]
        
        self.image_ind = 0
        
        if self.first_run:
            self.canvas_image = self.graph_canvas.create_image(
                0, 0, image=self.graph_images[0],  anchor='nw')
            self.update_recursive()
        else:
            self.graph_canvas.itemconfigure(self.canvas_image, image=self.graph_images[0])
        
    def bind_keyboard(self):
        def previous_image(event):
            self.image_ind = (self.image_ind - 1) % self.image_amount
            self.graph_canvas.itemconfigure(self.canvas_image, image=self.graph_images[self.image_ind])
        def next_image(event):
            self.image_ind = (self.image_ind + 1) % self.image_amount
            self.graph_canvas.itemconfigure(self.canvas_image, image=self.graph_images[self.image_ind])
        
        self.bind('<Left>', previous_image)
        self.bind('<Left>', next_image)
    
    
    def add_output(self, inp='', new_line=True):
        self.output += str(inp)
        if new_line: self.output += '\n'
                                           
    
    def run(self):
        
        self.output = ''
        
        window_size = 10
        
        self.polynomial_var = {
            'chebyshev_first': 'T',
            'chebyshev_second': 'U',
            'legendre': 'P',
            'hermite': 'H',
            'laguerre': 'L'
        }.get(self.polynomial_type.get())
        
        selected_var = self.y_coord.get()
        
        feature_amount = 3
        feature_filenames = (self.feature_filenames.get().format(feature_ind) for feature_ind in range(1, 1 + feature_amount))
        
        (x, feature_lengths), y = read_input_from_files(feature_filenames, self.label_filename.get())
    
        polynomial_degree_values = np.array([polynomial_degree.get() for polynomial_degree in self.polynomial_degrees])
        
        mult = self.mult.get()
        polynomial_type = self.polynomial_type.get()
               
        x = normalize_data(x)
        y, y_norm_values = normalize_data(y, min_max_data = True)
        
        x_features = split_data_by_features(x, feature_lengths)
        
        x_variable = x_features[selected_var - 1] 
        y_variable = y[:, selected_var - 1]
        
        y_windowed = timeseries_to_fixed_window_array_padded(y_variable, window_size=window_size)
           
        A_x = create_equation_matrix(
            x_variable, polynomial_type=polynomial_type, 
            polynomial_degree=polynomial_degree_values[selected_var - 1])
        
        A_y = create_equation_matrix(
            y_windowed, polynomial_type=polynomial_type, 
            polynomial_degree=polynomial_degree_values[selected_var - 1])
        
        A_x = normalize_data(A_x)
        A_y = normalize_data(A_y)
        
        A = concat_equation_matrices([A_x, A_y])
        
        lambda_matrix = solve(A, y_variable, mult=mult)
            
        err = np.max(np.abs(forward(A, lambda_matrix, mult=mult) - y_variable))
        
        '''while err > 0.5:
            
            polynomial_degree_values[selected_var - 1] += 1
            
            A = create_equation_matrix(
                x, polynomial_type=polynomial_type, 
                polynomial_degree=polynomial_degree_values[selected_var - 1])

            A = normalize_data(A)
            lambda_matrix = solve(A, y_variable, mult=mult)

            err = np.max(np.abs(forward(A, lambda_matrix, mult=mult) - y_variable))
            
            
        for polynomial_degree, value in zip(self.polynomial_degrees, polynomial_degree_values):
            polynomial_degree.set(value)'''
        
        approx_values = forward(A, lambda_matrix, mult=mult)
                
        if not self.normalize_y.get():
            y_variable = denormalize_data(y_variable, norm_values)
            approx_values = denormalize_data(approx_values, norm_values)
        
        self.image_amount = save_graph_sequence(y_variable, approx_values, step=100) 
        self.update_graph()
        
        err_value = np.max(np.abs(y_variable - approx_values))
        self.err_output.set('Значення похибки: {0:.5}'.format(err_value))
                
        self.set_output_text(self.output)
        
        if self.first_run: self.first_run = False