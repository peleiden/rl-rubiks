import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { HttpClientModule } from "@angular/common/http";

import { AppComponent } from './app.component';
import { RubiksComponent } from './rubiks/rubiks.component';
import { SideComponent } from './rubiks/side.component';

@NgModule({
  declarations: [
    AppComponent,
    RubiksComponent,
    SideComponent,
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
