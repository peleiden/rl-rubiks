import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { HttpClientModule } from "@angular/common/http";
import { FormsModule } from "@angular/forms";

import { AppComponent } from './app.component';
import { AppRoutingModule } from './app-routing.module';
import { CubeComponent } from './cube/cube.component';
import { AppendixComponent } from './appendix/appendix.component';
import { DocumentationComponent } from './documentation/documentation.component';
import { MethodsComponent } from './methods/methods.component';

@NgModule({
  declarations: [
    AppComponent,
    CubeComponent,
    AppendixComponent,
    DocumentationComponent,
    MethodsComponent,
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    FormsModule,
    AppRoutingModule,
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
